global.__base = __dirname + '/';

const
    use_https     = true,
    argv          = require('minimist')(process.argv.slice(2)),
    https         = require('https'),
    fs            = require('fs'),
    app           = require('express')(),
    _             = require('lodash'),
    parser        = require('xmldom').DOMParser,
    XMLHttpRequest = require("xmlhttprequest").XMLHttpRequest,
    sendPostRequest = require('request').post;


const researchers = ['A4SSYO0HDVD4E', 'A1BOIDKD33QSDK', 'A1MMCS8S8CTWKU','A1MMCS8S8CTWKV','A1MMCS8S8CTWKS', 'A1KXXBD1M6NBK5'];
const blockResearcher = false;

let gameport;
if(argv.gameport) {
  gameport = argv.gameport;
  console.log('using port ' + gameport);
} else {
  gameport = 8887;
  console.log('no gameport specified: using 8887\nUse the --gameport flag to change');
}

let server;
let io;
try {
  const privateKey  = fs.readFileSync('/etc/apache2/ssl/stanford-cogsci.org.key'),
        certificate = fs.readFileSync('/etc/apache2/ssl/stanford-cogsci.org.crt'),
        intermed    = fs.readFileSync('/etc/apache2/ssl/intermediate.crt'),
        options     = {key: privateKey, cert: certificate, ca: intermed};
  server            = require('https').createServer(options,app).listen(gameport),
  io                = require('socket.io')(server);
} catch (err) {
  console.log("cannot find SSL certificates; falling back to http");
  server = app.listen(gameport),
  io     = require('socket.io')(server);
}

// serve stuff that the client requests
app.get('/*', (req, res) => {
  const id = req.query.workerId;
  const isResearcher = _.includes(researchers, id);

  if(!id || id === 'undefined' || (isResearcher && !blockResearcher)) {

    // Let through if researcher, or in 'testing' mode
    serveFile(req, res);

  } else if(!valid_id(id)) {

    // If invalid id, block them
    console.log('invalid id, blocked');
    return handleInvalidID(req, res);

  } else {

    // If the database shows they've already participated, block them.
    console.log('neither invalid nor blank id, check if repeat worker');
    checkPreviousParticipant(id, (exists) => {
      return exists ? handleDuplicate(req, res) : serveFile(req, res);
    });
  }
});

io.on('connection', function (socket) {

  // Recover query string information and set condition
  const query = socket.handshake.query;

  // Send client stims
  initializeWithTrials(socket);

  // Set up callback for writing client data to mongo
  socket.on('currentData', function(data) {
    console.log('currentData received: ' + JSON.stringify(data));
    sendPostRequest(
      'http://localhost:6004/db/insert',
      { json: data },
      (error, res, body) => {
        if (!error && res.statusCode === 200) {
          console.log(`sent data to store`);
        } else {
	  console.log(`error sending data to store: ${error} ${body}`);
        }
      }
    );
  });
});

const serveFile = function(req, res) {
  const fileName = req.params[0];
  console.log('\t :: Express :: file requested: ' + fileName);
  return res.sendFile(fileName, {root: __dirname});
};

const handleDuplicate = function(req, res) {
  console.log("duplicate id: blocking request");
  res.sendFile('duplicate.html', {root: __dirname});
  return res.redirect('/duplicate.html');

};

const valid_id = function(id) {
  return (id.length <= 15 && id.length >= 12) || id.length == 41;
};

const handleInvalidID = function(req, res) {
  console.log("invalid id: blocking request");
  return res.redirect('/invalid.html');
};

function checkPreviousParticipant (workerId, callback) {
  const p = {'wID': workerId};
  const postData = {
    dbname: 'neural_constructions',
    query: p,
    projection: {'_id': 1}
  };
  sendPostRequest(
    'http://localhost:6004/db/exists',
    {json: postData},
    (error, res, body) => {
      try {
        if (!error && res.statusCode === 200) {
          console.log("success! Received data " + JSON.stringify(body));
          callback(body);
        } else {
          throw `${error}`;
        }
      }
      catch (err) {
        console.log(err);
        console.log('no database; allowing participant to continue');
        return callback(false);
      }
    }
  );
};

function initializeWithTrials(socket) {
  const gameid = UUID();
  sendPostRequest('http://localhost:6004/db/getstims', {
    json: {
      dbname: 'neural_constructions',
      colname: 'stimuli',
      gameid: gameid
    }
  }, (error, res, body) => {
    if (!error && res.statusCode === 200) {
      // send trial list (and id) to client
      console.log(body);
      var packet = _.extend({}, _.omit(body, ["_id", "numGames", "games"]), {
      	gameid: gameid
      });
      console.log('got condition packet from db: ', packet);
      socket.emit('onConnected', packet);
    } else {
      console.log(`error getting stims: ${error} ${body}`);
      console.log('returning hard-coded sticks');
      socket.emit('onConnected', {
	gameid: gameid,
	trials: _.clone(require('./example_trials.json'))
      });
    }
  });
}

function UUID () {
  var baseName = (Math.floor(Math.random() * 10) + '' +
        Math.floor(Math.random() * 10) + '' +
        Math.floor(Math.random() * 10) + '' +
        Math.floor(Math.random() * 10));
  var template = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx';
  var id = baseName + '-' + template.replace(/[xy]/g, function(c) {
    var r = Math.random()*16|0, v = c == 'x' ? r : (r&0x3|0x8);
    return v.toString(16);
  });
  return id;
};
