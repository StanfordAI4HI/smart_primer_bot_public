TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`

kill `cat gunicorn.pid`
mv logs/access.log logs/access_$TIMESTAMP.log
mv logs/error.log logs/error_$TIMESTAMP.log
