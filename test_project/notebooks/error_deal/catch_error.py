from osgeo import gdal

def gdal_error_handler(err_class, err_num, err_msg):
    err_type = {
        gdal.CE_None: 'None',
        gdal.CE_Debug: 'Debug',
        gdal.CE_Warning: 'Warning',
        gdal.CE_Failure: 'Failure',
        gdal.CE_Fatal: 'Fatal'
    }
    print(f"<<GDAL {err_type.get(err_class)}>> 编号: {err_num}, 描述: {err_msg}")