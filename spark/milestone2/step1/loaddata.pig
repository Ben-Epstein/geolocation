data = LOAD '/loudacre/data/devicestatus.txt' USING PigStorage(',') AS (date:chararray, model:chararray, device_id:chararray, fourth, fifth, sixth, seventh, eighth, ninth, tenth, eleventh, twelth, latitude:double, longitude:double);
parsed = filter data by TRIM(model)!='';
newrelation = FOREACH parsed GENERATE date, model, latitude, longitude;
latlong = FILTER newrelation BY (latitude != 0 AND longitude !=0);
splitrelation = FOREACH latlong GENERATE date,FLATTEN( STRSPLIT(model, ' ', 2)) as (manufacturer: chararray, model: chararray), latitude, longitude;
STORE splitrelation INTO '/loudacre/devicestaus_etl' USING PigStorage(',');
