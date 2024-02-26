var express = require('express')

var STATIC_FOLDER = "/home/allquill/docs/images";

var app = express()
app.use(express.static(STATIC_FOLDER));

const port = process.env.PORT || 3002;
console.log('App listening on port ' + port + '!');
app.listen(port);
