<?xml version="1.0" encoding="UTF-8"?>
<StyledLayerDescriptor xmlns="http://www.opengis.net/sld" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:se="http://www.opengis.net/se" xmlns:xlink="http://www.w3.org/1999/xlink" xsi:schemaLocation="http://www.opengis.net/sld http://schemas.opengis.net/sld/1.1.0/StyledLayerDescriptor.xsd" version="1.1.0" xmlns:ogc="http://www.opengis.net/ogc">
  <NamedLayer>
    <se:Name>EMSR756_AOI08_DEL_MONIT01_floodDepthA_v1</se:Name>
    <UserStyle>
      <se:Name>EMSR756_AOI08_DEL_MONIT01_floodDepthA_v1</se:Name>
      <se:FeatureTypeStyle>
	  
        <se:Rule>
          <se:Name>Below 0.50</se:Name>
          <se:Description>
            <se:Title>Below 0.50</se:Title>
          </se:Description>
          <ogc:Filter xmlns:ogc="http://www.opengis.net/ogc">
            <ogc:PropertyIsEqualTo>
			  <ogc:PropertyName>value</ogc:PropertyName>
             <ogc:Literal>Below 0.50</ogc:Literal>
            </ogc:PropertyIsEqualTo>
          </ogc:Filter>
          <se:PolygonSymbolizer>
            <se:Fill>
              <se:SvgParameter name="fill">#66CCAA</se:SvgParameter>
            </se:Fill>
          </se:PolygonSymbolizer>
        </se:Rule>

          <se:Rule>
          <se:Name>0.50 - 1.00</se:Name>
          <se:Description>
            <se:Title>0.50 - 1.00</se:Title>
          </se:Description>
          <ogc:Filter xmlns:ogc="http://www.opengis.net/ogc">
            <ogc:PropertyIsEqualTo>
			  <ogc:PropertyName>value</ogc:PropertyName>
             <ogc:Literal>0.50 - 1.00</ogc:Literal>
            </ogc:PropertyIsEqualTo>
          </ogc:Filter>
          <se:PolygonSymbolizer>
            <se:Fill>
              <se:SvgParameter name="fill">#50BAB0</se:SvgParameter>
            </se:Fill>
          </se:PolygonSymbolizer>
        </se:Rule>

          <se:Rule>
          <se:Name>1.00 - 2.00</se:Name>
          <se:Description>
            <se:Title>1.00 - 2.00</se:Title>
          </se:Description>
          <ogc:Filter xmlns:ogc="http://www.opengis.net/ogc">
            <ogc:PropertyIsEqualTo>
			  <ogc:PropertyName>value</ogc:PropertyName>
             <ogc:Literal>1.00 - 2.00</ogc:Literal>
            </ogc:PropertyIsEqualTo>
          </ogc:Filter>
          <se:PolygonSymbolizer>
            <se:Fill>
              <se:SvgParameter name="fill">#3C98AB</se:SvgParameter>
            </se:Fill>
          </se:PolygonSymbolizer>
        </se:Rule>

          <se:Rule>
          <se:Name>2.00 - 4.00</se:Name>
          <se:Description>
            <se:Title>2.00 - 4.00</se:Title>
          </se:Description>
          <ogc:Filter xmlns:ogc="http://www.opengis.net/ogc">
            <ogc:PropertyIsEqualTo>
			  <ogc:PropertyName>value</ogc:PropertyName>
             <ogc:Literal>2.00 - 4.00</ogc:Literal>
            </ogc:PropertyIsEqualTo>
          </ogc:Filter>
          <se:PolygonSymbolizer>
            <se:Fill>
              <se:SvgParameter name="fill">#296999</se:SvgParameter>
            </se:Fill>
          </se:PolygonSymbolizer>
        </se:Rule>

          <se:Rule>
          <se:Name>4.00 - 6.00</se:Name>
          <se:Description>
            <se:Title>4.00 - 6.00</se:Title>
          </se:Description>
          <ogc:Filter xmlns:ogc="http://www.opengis.net/ogc">
            <ogc:PropertyIsEqualTo>
			  <ogc:PropertyName>value</ogc:PropertyName>
             <ogc:Literal>4.00 - 6.00</ogc:Literal>
            </ogc:PropertyIsEqualTo>
          </ogc:Filter>
          <se:PolygonSymbolizer>
            <se:Fill>
              <se:SvgParameter name="fill">#1A3B87</se:SvgParameter>
            </se:Fill>
          </se:PolygonSymbolizer>
        </se:Rule>

		</se:FeatureTypeStyle>
	  
    </UserStyle>
  </NamedLayer>
</StyledLayerDescriptor>
