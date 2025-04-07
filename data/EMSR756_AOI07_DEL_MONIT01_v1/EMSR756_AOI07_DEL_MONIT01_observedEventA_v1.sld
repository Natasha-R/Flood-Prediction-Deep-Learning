<StyledLayerDescriptor xmlns="http://www.opengis.net/sld" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:se="http://www.opengis.net/se" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:ogc="http://www.opengis.net/ogc" xsi:schemaLocation="http://www.opengis.net/sld http://schemas.opengis.net/sld/1.1.0/StyledLayerDescriptor.xsd" version="1.1.0">
  <NamedLayer>
    <se:Name>EMSR756_AOI07_DEL_MONIT01_observedEventA_v1</se:Name>
    <UserStyle>
      <se:Name>EMSR756_AOI07_DEL_MONIT01_observedEventA_v1</se:Name>
      <se:FeatureTypeStyle>
	  
        <se:Rule>
          <se:Name>Flooded area</se:Name>
          <se:Description>
            <se:Title>Flooded area</se:Title>
          </se:Description>
          <ogc:Filter xmlns:ogc="http://www.opengis.net/ogc">
            <ogc:PropertyIsEqualTo>
			  <ogc:PropertyName>notation</ogc:PropertyName>
             <ogc:Literal>Flooded area</ogc:Literal>
            </ogc:PropertyIsEqualTo>
          </ogc:Filter>
		  
          <se:PolygonSymbolizer>
            <se:Fill>
              <se:SvgParameter name="fill">#00c5ff</se:SvgParameter>
            </se:Fill>
          </se:PolygonSymbolizer>
        </se:Rule>

        </se:FeatureTypeStyle>
	  
	  
	  
	  
	  
	  
	  
    </UserStyle>
  </NamedLayer>
</StyledLayerDescriptor>
