"""Parser for the FITS format, as used in astronomy.

A FITS file is a sequence of HDUs (Header-Data Units), each a text header
followed by a contiguous, uncompressed, big-endian data block. Astropy parses
the headers -- which need real fixing-up of non-conforming cards -- and reports
where each data block starts, which is exactly what a ChunkManifest needs.

See the FITS Standard, version 4.0.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from packaging.version import Version

from virtualizarr.codecs import FITS_ASCII_CODEC_NAME
from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.parsers.utils import encode_cf_fill_value

# FITS stores every value big-endian, so BITPIX only has to choose the width and
# whether the value is an integer (positive) or IEEE float (negative).
_BITPIX2DTYPE: dict[int, str] = {
    8: "uint8",
    16: ">i2",
    32: ">i4",
    64: ">i8",
    -32: ">f4",
    -64: ">f8",
}

# The header keyword holding a card's free text, which has no single value to
# record as an attribute.
_COMMENTARY_KEYWORDS = ("COMMENT", "HISTORY", "")

# The bytes codec only applies its `endian` to the fields of a structured dtype from
# this release on (zarr-python#4142). Below it, a binary table's big-endian columns
# decode as though they were little-endian, and say nothing about it.
_STRUCT_ENDIAN_FIX = Version("3.3.0")

# TFORM for a variable-length column: an optional repeat count, then P or Q.
_VARIABLE_LENGTH_FORMAT = re.compile(r"\d*[PQ]")


def _attributes(header: Any) -> dict[str, Any]:
    """Convert a FITS header into JSON-serializable Zarr attributes."""
    return {
        key: value if isinstance(value, (bool, int, float, str)) else str(value)
        for key, value in dict(header).items()
        if key not in _COMMENTARY_KEYWORDS
    }


def _dimension_names(name: str, ndim: int) -> tuple[str, ...]:
    """Name an HDU's axes, which FITS itself leaves anonymous.

    Names are qualified by the HDU they belong to, because separate HDUs in one
    file routinely disagree about the length of what would otherwise be the same
    axis, and xarray allows a dimension name only one length.
    """
    if ndim == 1:
        return (f"{name}_row",)
    if ndim <= 3:
        # The conventional axis names for an image or cube.
        return tuple(f"{name}_{axis}" for axis in ["z", "y", "x"][-ndim:])
    return tuple(f"{name}_axis{ndim - i}" for i in range(ndim))


def _image_array(hdu: Any, name: str, url: str) -> ManifestArray:
    """Build a ManifestArray for an image or cube HDU.

    ``BSCALE``/``BZERO`` make the stored integers a scaled representation of the
    physical values, so the array becomes float and the stored dtype moves into a
    scale-offset codec.
    """
    naxis = hdu.header["NAXIS"]
    # NAXIS1 varies fastest, so the Zarr shape counts the axes back down.
    shape = tuple(int(hdu.header[f"NAXIS{i}"]) for i in range(naxis, 0, -1))
    stored_dtype = np.dtype(_BITPIX2DTYPE[hdu.header["BITPIX"]])
    nbytes = stored_dtype.itemsize * math.prod(shape)

    attributes = _attributes(hdu.header)
    # BLANK marks undefined pixels of an integer image. Floating-point images use
    # IEEE NaN instead and have no BLANK.
    blank = hdu.header.get("BLANK") if stored_dtype.kind in "iu" else None

    codecs: list[dict[str, Any]] | None = None
    dtype = stored_dtype
    if "BSCALE" in hdu.header or "BZERO" in hdu.header:
        bscale = float(hdu.header.get("BSCALE", 1))
        if bscale == 0:
            raise ValueError(
                f"Image {name!r} declares BSCALE=0, which would make every stored "
                "value decode to the same physical value."
            )
        if blank is not None:
            # BLANK is a stored value, matched before scaling, but the array it
            # labels is the scaled one, so the sentinel has to be scaled to match.
            blank = float(hdu.header.get("BZERO", 0)) + bscale * blank
        dtype = np.dtype("float64")
        codecs = [
            {
                "name": "numcodecs.fixedscaleoffset",
                "configuration": {
                    "offset": float(hdu.header.get("BZERO", 0)),
                    # FITS decodes as BZERO + BSCALE * stored, but the codec divides
                    # by scale rather than multiplying, so it takes the reciprocal.
                    "scale": 1 / bscale,
                    "astype": stored_dtype.str,
                    "dtype": dtype.str,
                },
            },
            # The array dtype is now the decoded float, so the bytes codec has to be
            # spelled out to keep reading the big-endian stored values.
            {"name": "bytes", "configuration": {"endian": "big"}},
        ]

    if blank is not None:
        # A sentinel within the data rather than a marker of unwritten storage, so it
        # belongs in the CF attribute that masks values, not in the Zarr fill value.
        attributes["_FillValue"] = encode_cf_fill_value(np.array(blank, dtype), dtype)

    metadata = create_v3_array_metadata(
        shape=shape,
        data_type=dtype,
        chunk_shape=shape,
        codecs=codecs,
        attributes=attributes,
        dimension_names=_dimension_names(name, len(shape)),
    )
    return _single_chunk_array(metadata, hdu, url, nbytes)


def _ascii_table_array(hdu: Any, name: str, url: str) -> ManifestArray:
    """Build a ManifestArray for an ASCII table HDU.

    Columns are stored as fixed-width text, so the values are recovered by a codec
    that reads each column's characters and casts them to the declared type.
    """
    names = hdu.columns.names
    # Width in characters of each column's text field.
    spans = hdu.columns._spans
    # TBCOLn is where column n starts within a row, counting from 1. It is what
    # actually places the columns: FITS permits gaps between them, so their widths
    # cannot be assumed to tile the row.
    starts = [int(hdu.header[f"TBCOL{i + 1}"]) - 1 for i in range(len(names))]
    dtypes = [hdu.columns[name].format.recformat for name in names]
    columns = [list(column) for column in zip(names, starts, spans, dtypes)]

    nrows = int(hdu.header["NAXIS2"])
    row_nbytes = int(hdu.header["NAXIS1"])
    nbytes = row_nbytes * nrows

    metadata = create_v3_array_metadata(
        shape=(nrows,),
        data_type=np.dtype(list(zip(names, dtypes))),
        chunk_shape=(nrows,),
        codecs=[
            {
                "name": FITS_ASCII_CODEC_NAME,
                "configuration": {"columns": columns, "row_nbytes": row_nbytes},
            }
        ],
        attributes=_attributes(hdu.header),
        dimension_names=_dimension_names(name, 1),
    )
    return _single_chunk_array(metadata, hdu, url, nbytes)


def _bintable_array(hdu: Any, name: str, url: str) -> ManifestArray:
    """Build a ManifestArray for a binary table HDU.

    A binary table's row is a C struct, which maps onto the Zarr v3 ``struct`` data
    type. That data type records no byte order -- the fields reparse native -- but
    byte order is the bytes codec's to carry, and ``convert_to_codec_pipeline`` reads
    it off the dtype and writes ``endian: big``, as FITS always stores a table.

    Columns holding a fixed-length array survive as raw bytes rather than as numbers:
    the v3 struct cannot express a subarray field, so ``('>f4', (5,))`` becomes an
    opaque ``raw_bytes`` field of the same width. The bytes are the stored ones, in
    the file's own byte order, so a reader recovers them with
    ``np.ascontiguousarray(table["SPECTROFLUX"]).view(">f4")``.
    """
    if Version(zarr.__version__) < _STRUCT_ENDIAN_FIX:
        raise ValueError(
            f"Binary table {name!r} needs zarr >= {_STRUCT_ENDIAN_FIX} to read "
            f"correctly (this is zarr {zarr.__version__}). The Zarr v3 'struct' data "
            "type records no byte order, so a FITS table's big-endian columns rely on "
            "the bytes codec, which only byte-swaps a structured dtype's fields from "
            "that release (zarr-python#4142); older zarr returns byte-swapped numbers "
            f"without complaint. Upgrade zarr, or pass skip_variables=['{name}']."
        )

    # A variable-length column stores a (count, offset) descriptor pointing into the
    # heap that follows the table. The heap is outside the byte range this HDU's one
    # chunk covers, so those columns cannot be served at all -- and unlike a
    # fixed-length array column, their bytes would read back as plausible small
    # integers rather than as the values they point at.
    variable = [
        column.name
        for column in hdu.columns
        if _VARIABLE_LENGTH_FORMAT.match(str(column.format))
    ]
    if variable:
        raise ValueError(
            f"Binary table {name!r} has variable-length columns {variable}, whose "
            "values live in the heap after the table rather than in the table itself. "
            "A chunk manifest addresses one contiguous range, which cannot reach the "
            f"heap, so those columns would read back as descriptors. Pass "
            f"skip_variables=['{name}'] to exclude it."
        )

    stored_dtype = hdu.columns.dtype.newbyteorder(">")
    nrows = int(hdu.header["NAXIS2"])
    row_nbytes = int(hdu.header["NAXIS1"])

    metadata = create_v3_array_metadata(
        shape=(nrows,),
        data_type=stored_dtype,
        chunk_shape=(nrows,),
        # `Struct.default_scalar()` casts the integer 0 into every field, which a
        # `raw_bytes` field rejects, so the zeroed row is spelled out here instead.
        fill_value=np.zeros(1, stored_dtype)[0],
        attributes=_attributes(hdu.header),
        dimension_names=_dimension_names(name, 1),
    )
    return _single_chunk_array(metadata, hdu, url, row_nbytes * nrows)


def _single_chunk_array(
    metadata: Any, hdu: Any, url: str, nbytes: int
) -> ManifestArray:
    """Point a whole HDU at its one contiguous data block.

    FITS never chunks or compresses a data block, so the entire HDU is a single
    chunk beginning at the offset astropy reports for it.
    """
    ndim = len(metadata.shape)
    key = ".".join(["0"] * ndim) or "0"
    manifest = ChunkManifest(
        entries={
            key: {"path": url, "offset": hdu.fileinfo()["datLoc"], "length": nbytes}
        },
        shape=(1,) * ndim,
    )
    return ManifestArray(metadata=metadata, chunkmanifest=manifest)


def _build_manifest_array(hdu: Any, name: str, url: str) -> ManifestArray:
    """Build a ManifestArray for a single HDU, dispatching on its type."""
    from astropy.io import fits

    if hdu.is_image:
        return _image_array(hdu, name, url)
    if isinstance(hdu, fits.hdu.table.TableHDU):
        return _ascii_table_array(hdu, name, url)
    if isinstance(hdu, fits.hdu.table.BinTableHDU):
        return _bintable_array(hdu, name, url)
    raise ValueError(
        f"HDU {name!r} has unsupported type {type(hdu).__name__}. Pass "
        f"skip_variables=['{name}'] to exclude it."
    )


def _hdu_names(hdulist: Any) -> list[str]:
    """Name each HDU, keeping names unique across the file.

    FITS lets several HDUs share an ``EXTNAME`` (an image and its error and quality
    planes are often all repeated per exposure), but Zarr array names within a group
    must be distinct, so a repeat is qualified by its extension number.
    """
    names: list[str] = []
    for index, hdu in enumerate(hdulist):
        name = hdu.name or str(index)
        if name in names:
            name = f"{name}_{index}"
        names.append(name)
    return names


class FITSParser:
    """Create a [ManifestStore][virtualizarr.manifests.ManifestStore] from a FITS file.

    Every HDU holding data becomes an array: images and cubes of any rank, and
    ASCII tables, and binary tables. A binary table needs zarr >= 3.3.0, where the
    bytes codec byte-swaps a structured dtype's fields; its fixed-length array
    columns arrive as raw bytes, because the Zarr v3 ``struct`` data type cannot
    express a subarray field. A table with variable-length columns raises unless
    skipped: their values live in the heap, outside the range a chunk addresses.

    Parameters
    ----------
    group
        The group within the file to be used as the Zarr root group for the ManifestStore.
        FITS files are flat, so only the root group exists.
    skip_variables
        Variables in the file that will be ignored when creating the ManifestStore.
    """

    def __init__(
        self,
        group: str | None = None,
        skip_variables: Iterable[str] | None = None,
    ):
        self.group = group
        self.skip_variables = skip_variables

    def __call__(
        self,
        url: str,
        registry: ObjectStoreRegistry,
    ) -> ManifestStore:
        """
        Parse the metadata and byte offsets from a given FITS file to produce a VirtualiZarr ManifestStore.

        Parameters
        ----------
        url
            The URL of the input FITS file (e.g., "s3://bucket/file.fits").
        registry
            An [ObjectStoreRegistry][obspec_utils.registry.ObjectStoreRegistry] for resolving urls and reading data.

        Returns
        -------
        ManifestStore
            A ManifestStore which provides a Zarr representation of the parsed FITS file.
        """
        from astropy.io import fits
        from obspec_utils.readers import BlockStoreReader

        if self.group not in (None, "", "/"):
            raise ValueError(
                f'FITS files contain only a root group, so group="{self.group}" cannot be opened'
            )

        store, path_in_store = registry.resolve(url)
        reader = BlockStoreReader(store=store, path=path_in_store)

        skip = set(self.skip_variables or ())
        arrays = {}
        attributes = {}
        # Headers are read eagerly but data blocks are not, so only the headers and
        # whichever blocks they happen to share a buffered block with get fetched.
        with fits.open(reader, do_not_scale_image_data=True) as hdulist:
            names = _hdu_names(hdulist)
            for index, (name, hdu) in enumerate(zip(names, hdulist)):
                if name in skip:
                    continue
                if hdu.header.get("NAXIS", 0) == 0:
                    # An HDU with no data axes carries only metadata. The primary one
                    # conventionally describes the whole file, so it becomes the root
                    # group's attributes.
                    if index == 0:
                        attributes = _attributes(hdu.header)
                    continue
                # Rendering the header fixes up any non-conforming cards, which has to
                # happen before the values are read back out of it.
                str(hdu.header)
                arrays[name] = _build_manifest_array(hdu, name, url)

        manifest_group = ManifestGroup(arrays=arrays, attributes=attributes)
        return ManifestStore(group=manifest_group, registry=registry)
