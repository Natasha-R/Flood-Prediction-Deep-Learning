<StyledLayerDescriptor xmlns="http://www.opengis.net/sld" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:se="http://www.opengis.net/se" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:ogc="http://www.opengis.net/ogc" xsi:schemaLocation="http://www.opengis.net/sld http://schemas.opengis.net/sld/1.1.0/StyledLayerDescriptor.xsd" version="1.1.0">
  <NamedLayer>
    <se:Name>EMSR756_AOI08_DEL_PRODUCT_observedEventA_v2</se:Name>
    <UserStyle>
      <se:Name>EMSR756_AOI08_DEL_PRODUCT_observedEventA_v2</se:Name>
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

        <se:Rule>
          <se:Name>Flood trace</se:Name>
          <se:Description>
            <se:Title>Flood trace</se:Title>
          </se:Description>
          <ogc:Filter xmlns:ogc="http://www.opengis.net/ogc">
            <ogc:PropertyIsEqualTo>
              <ogc:PropertyName>notation</ogc:PropertyName>
              <ogc:Literal>Flood trace</ogc:Literal>
            </ogc:PropertyIsEqualTo>
          </ogc:Filter>
		  
          <se:PolygonSymbolizer>
            <se:Fill>
              <se:GraphicFill>
                <se:Graphic>
                  <se:ExternalGraphic>
                  <se:OnlineResource xlink:href="https://emergency.copernicus.eu/images/svg/observed_event_polygon_fill_flood_trace.svg" xlink:type="simple"/>
                  <se:Format>image/svg+xml</se:Format>
               </se:ExternalGraphic>
                  <se:Size>24</se:Size>
                </se:Graphic>
              </se:GraphicFill>
            </se:Fill>
          </se:PolygonSymbolizer>

		  
          <se:LineSymbolizer>
            <se:Stroke>
              <se:SvgParameter name="stroke">#00dca9</se:SvgParameter>
              <se:SvgParameter name="stroke-width">2</se:SvgParameter>
              <se:SvgParameter name="stroke-linejoin">bevel</se:SvgParameter>
              <se:SvgParameter name="stroke-linecap">square</se:SvgParameter>
            </se:Stroke>
          </se:LineSymbolizer>
        </se:Rule>

        </se:FeatureTypeStyle>
	  
	  
	  
	  
	  
	  
	  
    </UserStyle>
  </NamedLayer>
</StyledLayerDescriptor>
