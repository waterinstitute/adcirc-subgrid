from shapely.geometry import Polygon

from AdcircSubgrid.raster_region import RasterRegion


def test_raster_region() -> None:
    """
    Test the RasterRegion class
    """
    region = RasterRegion(
        geo_transform=[0.0, 1.0, 0.0, 10.0, 0.0, -1.0],
        x_size=10,
        y_size=10,
        x1=1.0,
        y1=2.0,
        x2=3.0,
        y2=5.0,
    )

    assert region.valid()
    assert not region.clamped()
    assert region.i_size() == 2
    assert region.j_size() == 3
    assert region.xll() == 1.0
    assert region.yll() == 2.0
    assert region.xur() == 3.0
    assert region.yur() == 5.0
    assert region.i_start() == 1
    assert region.j_start() == 5
    assert region.i_end() == 3
    assert region.j_end() == 8
    assert region.polygon() == Polygon([(1.0, 2.0), (3.0, 2.0), (3.0, 5.0), (1.0, 5.0)])
