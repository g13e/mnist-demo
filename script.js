//ONNX.JS code based on the original tutorial from MS here: https://github.com/microsoft/onnxjs-demo/tree/master
//Canvas drawing code based on dev.opera tutorial here: https://dev.opera.com/articles/html5-canvas-painting/

// Keep everything in anonymous function, called on window load.
if(window.addEventListener) {
window.addEventListener('load', function () {
  var canvas, cvsScaled,cvsCenterCrop;
  var context, ctxScaled,ctxCenterCrop;
  var tool,session;

  function init () {
    // Find the canvas element.
    canvas = document.getElementById('imageView');
    cvsScaled= document.getElementById('input-canvas-scaled');
    cvsCenterCrop= document.getElementById('input-canvas-centercrop')

    // Get the 2D canvas context.
    context = canvas.getContext('2d');
    ctxScaled = cvsScaled.getContext('2d');
    ctxCenterCrop = cvsCenterCrop.getContext('2d');

    //set context style
    context.lineWidth = 20;

    // Pencil tool instance.
    tool = new tool_pencil();

    //attach an event listener to the prediction button
    pred_bt=document.getElementById("pred_bt");
    pred_bt.addEventListener('click',predict,false);

    clear_bt=document.getElementById("clear_bt");
    clear_bt.addEventListener('click',clear,false);

    // Attach the mousedown, mousemove and mouseup event listeners.
    canvas.addEventListener('mousedown', ev_canvas, false);
    canvas.addEventListener('mousemove', ev_canvas, false);
    canvas.addEventListener('mouseup',   ev_canvas, false);

    initNN();
  }

  // This painting tool works like a drawing pencil which tracks the mouse 
  // movements.
  function tool_pencil () {
    var tool = this;
    this.started = false;

    // This is called when you start holding down the mouse button.
    // This starts the pencil drawing.
    this.mousedown = function (ev) {
        context.beginPath();
        context.moveTo(ev._x, ev._y);
        tool.started = true;
    };

    // This function is called every time you move the mouse. Obviously, it only 
    // draws if the tool.started state is set to true (when you are holding down 
    // the mouse button).
    this.mousemove = function (ev) {
      if (tool.started) {
        context.lineTo(ev._x, ev._y);
        context.stroke();
      }
    };

    // This is called when you release the mouse button.
    this.mouseup = function (ev) {
      if (tool.started) {
        tool.mousemove(ev);
        tool.started = false;
      }
    };
  }

  // The general-purpose event handler. This function just determines the mouse 
  // position relative to the canvas element.
  function ev_canvas (ev) {
    if (ev.layerX || ev.layerX == 0) { // Firefox
      ev._x = ev.layerX;
      ev._y = ev.layerY;
    } else if (ev.offsetX || ev.offsetX == 0) { // Opera
      ev._x = ev.offsetX;
      ev._y = ev.offsetY;
    }

    // Call the event handler of the tool.
    var func = tool[ev.type];
    if (func) {
      func(ev);
    }
  }

  function clear(){
    context.clearRect(0, 0, canvas.width, canvas.height);
    ctxScaled.clearRect(0, 0, cvsScaled.width, cvsScaled.height);
    ctxCenterCrop.clearRect(0, 0, cvsCenterCrop.width, cvsCenterCrop.height);
  }
  
  function initNN(){
    // create a session
    session = new onnx.InferenceSession({backendHint: "webgl"});
    // load the ONNX model file
    // use the following in an async method
    const url = "./mnist.onnx";
    session.loadModel(url);
    console.log("Network initialized!");
  }

  async function predict(){
    var pred_div=document.getElementById("prediction");

    //define input
    //const inputs = [new Tensor(new Float32Array([1.0, 2.0, 3.0, 4.0]), "float32", [2, 2])];
    const tensor = getInput();

    //predict, run this in an async method
    // Run model with Tensor inputs and get the result by output name defined in model.
    const outputMap = await session.run([tensor]);
    console.log(outputMap);
    
    const outputTensor = outputMap.values().next().value;
    const output = getPredictedClass(outputTensor);
    console.log(output);
    
    number=output;
    var pred_content = document.createTextNode("maybe "+number+" ??");
    pred_div.innerHTML="";
    pred_div.appendChild(pred_content);
  }

  function getInput(){

    // center crop
    const imageDataCenterCrop = centerCrop(context.getImageData(0, 0, context.canvas.width, context.canvas.height));
    
    ctxCenterCrop.canvas.width = imageDataCenterCrop.width;
    ctxCenterCrop.canvas.height = imageDataCenterCrop.height;
    ctxCenterCrop.putImageData(imageDataCenterCrop, 0, 0);
   
    // scaled to 28 x 28
    //const ctxScaled = document.getElementById('input-canvas-scaled').getContext('2d');
    ctxScaled.save();
    ctxScaled.scale(28 / ctxCenterCrop.canvas.width, 28 / ctxCenterCrop.canvas.height);
    ctxScaled.clearRect(0, 0, ctxCenterCrop.canvas.width, ctxCenterCrop.canvas.height);
    ctxScaled.drawImage(document.getElementById('input-canvas-centercrop'), 0, 0);
    const imageDataScaled = ctxScaled.getImageData(0, 0, ctxScaled.canvas.width, ctxScaled.canvas.height);
    ctxScaled.restore();
    // process image data for model input
    const { data } = imageDataScaled;
    const input = new Float32Array(784);
    for (let i = 0, len = data.length; i < len; i += 4) {
      input[i / 4] = data[i + 3] / 255;
    }

    const tensor = new Tensor(input, 'float32', [1, 1, 28, 28]);
    return tensor;

  }

  function getPredictedClass(rawOutput) {
    //output=postprocess(rawOutput);
    output=rawOutput.data;
    if (output.reduce((a, b) => a + b, 0) === 0) { 
      return -1;
    }
    return output.reduce((argmax, n, i) => (n > output[argmax] ? i : argmax), 0);
  }

  function postprocess(rawOutput){
    return softmax(Array.prototype.slice.call(rawOutput.data));  
  }

  function softmax(arr) {
    const C = Math.max(...arr);
    const d = arr.map((y) => Math.exp(y - C)).reduce((a, b) => a + b);
    return arr.map((value, index) => { 
        return Math.exp(value - C) / d;
    });
  }

  function centerCrop(imageData) {
    const { data, width, height } = imageData;
    let [xmin, ymin] = [width, height];
    let [xmax, ymax] = [-1, -1];
    for (let i = 0; i < width; i++) {
      for (let j = 0; j < height; j++) {
        const idx = i + j * width;
        if (data[4 * idx + 3] > 0) {
          if (i < xmin) {xmin = i;}
          if (i > xmax) {xmax = i;}
          if (j < ymin) {ymin = j;}
          if (j > ymax) {ymax = j;}
        }
      }
    }
  
    // add a little padding
    xmin -= 20;
    xmax += 20;
    ymin -= 20;
    ymax += 20;
  
    // make bounding box square
    let [widthNew, heightNew] = [xmax - xmin + 1, ymax - ymin + 1];
    if (widthNew < heightNew) {
      // new width < new height
      const halfBefore = Math.floor((heightNew - widthNew) / 2);
      const halfAfter = heightNew - widthNew - halfBefore;
      xmax += halfAfter;
      xmin -= halfBefore;
    } else if (widthNew > heightNew) {
      // new width > new height
      const halfBefore = Math.floor((widthNew - heightNew) / 2);
      const halfAfter = widthNew - heightNew - halfBefore;
      ymax += halfAfter;
      ymin -= halfBefore;
    }
  
    widthNew = xmax - xmin + 1;
    heightNew = ymax - ymin + 1;
    const dataNew = new Uint8ClampedArray(widthNew * heightNew * 4);
    for (let i = xmin; i <= xmax; i++) {
      for (let j = ymin; j <= ymax; j++) {
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const idx = i + j * width;
          const idxNew = i - xmin + (j - ymin) * widthNew;
          dataNew[4 * idxNew + 3] = data[4 * idx + 3];
        }
      }
    }
  
    return new ImageData(dataNew, widthNew, heightNew);
  }

  init();

}, false); }
