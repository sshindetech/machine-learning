var express = require('express')

var app = express()

var STATIC_FOLDER = "/home/allquill/docs/images";

app.use(express.static(STATIC_FOLDER))