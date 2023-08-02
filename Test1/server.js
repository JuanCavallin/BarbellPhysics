const express = require("express");
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const Handlebars = require("handlebars");

var app = express();
app.use(cors()); // Allows incoming requests from any IP
//app.use(express.static(path.join(__dirname, "index.html")))

//Reference to Python script
const {spawn} = require('child_process')

// Start by creating some disk storage options:
const storage = multer.diskStorage({
    destination: function (req, file, callback) {
        callback(null, __dirname + '/uploads');
    },
    // Sets file(s) to be saved in uploads folder in same directory
    filename: function (req, file, callback) {
        callback(null, Date.now() + path.extname(file.originalname) );
    }
    // Sets saved filename(s) to be original filename(s)
  })
  
// Set saved storage options:
const upload = multer({ storage: storage })

app.get("/", (req, res) => {
    
    res.sendFile(path.join(__dirname, "index.html"))

    //Load python script as a test
    /**
     * var dataToSend
    const python = spawn('hello.py', ['analyze_squat.py'])
     // collect data from script
     python.stdout.on('data', function (data) {
        console.log('Pipe data from python script ...');
        dataToSend = data.toString();
    });

    python.stderr.on('data', function (data) {
        console.error('stderr:' + data)

    })
    // in close event we are sure that stream from child process is closed
    python.on('close', (code) => {
    console.log(`child process close all stdio with code ${code}`);
    // send data to browser
    
    res.send(dataToSend)
    });
     */
    
    
    //Add res.json videos succesfully displayed?
})

app.post("/upload", upload.array("files"), (req, res) => {
    
    // Sets multer to intercept files named "files" on uploaded form data
    console.log(req.body); // Logs form body values
    console.log(req.files); // Logs any files
    res.json({ message: "File(s) uploaded successfully" });
    //res.sendFile(path.join(__dirname, "video.html"));

    

});



app.listen(5000, function(){
    console.log("Server running on port 5000");
});