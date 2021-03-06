/*
WebotoonGAN
Copyright (c) 2021-present Hyunkwon Ko, Subin An

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
*/

// Refer to https://github.com/quolc/neural-collage/blob/master/static/demo_feature_blending/js/main.js

max_colors = 5;
colors = [
  "#FF3B30", 
  "#4Cd964",
  "#5856d6",
  "#FF9500",
  "#007AFF",
];
original_image = null;
original_choose = null;

palette = null;
mask_selected_index = 0;

ui_uninitialized = true;
p5_input_original = null;
p5_input_reference = null;
p5_output = null;
sync_flag = true;
id = null;
brush_size = 20;
sf = 1;
mask_thumnail_size = 55;
pose_lr = 0

function ReferenceNameSpace() {
  return function (s) {
    s.setup = function () {
      s.pixelDensity(1);
      s.createCanvas(canvas_size, canvas_size);

      s.mask = [];
      for (var i = 0; i < max_colors; i++) {
        s.mask.push(s.createGraphics(canvas_size, canvas_size));
      }

      s.body = null;
      s.cursor(s.HAND);
      s.rotate_angle = 0;
      s.rotate_mode = false;
    };

    s.draw = function () {
      // Image
      s.push();
      s.translate(canvas_size / 2, canvas_size / 2);
      s.imageMode(s.CENTER);
      // s.rotate(s.rotate_angle);
      s.scale(sf, 1);
      s.background(255);
      s.noTint();
      if (s.body != null) {
        s.image(s.body, 0, 0, s.width, s.height);
      }
      s.pop();
      // Mask
      s.push();
      s.translate(canvas_size / 2, canvas_size / 2);
      s.imageMode(s.CENTER);
      // s.scale(1, sf);
      s.tint(255, 127);
      if (mask_selected_index != null)
        s.image(s.mask[mask_selected_index], 0, 0);

      s.pop();
      s.push();
      s.fill("rgba(46, 49, 49, 0.4)");
      s.strokeWeight(0);
      s.ellipse(s.mouseX, s.mouseY, brush_size, brush_size);
      s.pop();
    };

    s.keyPressed = function () {
      // // keyboar r
      // if (s.keyCode === 82) {
      //   if (s.rotate_mode) $("#p5-reference").css({ outline: "none" });
      //   else $("#p5-reference").css({ outline: "solid" });
      //   s.rotate_mode = !s.rotate_mode;
      // }

      // // keyboard <>
      // if (s.rotate_mode & (s.keyCode === 188)) {
      //   s.rotate_angle -= s.PI / 8;
      // }

      // if (s.rotate_mode & (s.keyCode === 190)) {
      //   s.rotate_angle += s.PI / 8;
      // }

      // keyboard []
      if (s.keyCode === 219) {
        brush_size -= 5;
        brush_size = Math.max(brush_size, 5);
      }
      if (s.keyCode === 221) {
        brush_size += 5;
        brush_size = Math.min(brush_size, 100);
      }
    };

    s.mouseDragged = function () {
      if (ui_uninitialized) return;

      var c = $(".palette-item.selected").data("class");
      if (c != -1) {
        var col = s.color(colors[mask_selected_index]);
        s.mask[mask_selected_index].noStroke();
        s.mask[mask_selected_index].fill(col);
        s.mask[mask_selected_index].ellipse(
          s.mouseX,
          s.mouseY,
          brush_size,
          brush_size
        );
      } else {
        // eraser
        if (sync_flag == true) {
          var col = s.color(0, 0);
          erase_size = brush_size / 2;
          s.mask[mask_selected_index].loadPixels();
          for (
            var x = Math.max(0, Math.floor(s.mouseX) - erase_size);
            x < Math.min(canvas_size, Math.floor(s.mouseX) + erase_size);
            x++
          ) {
            for (
              var y = Math.max(0, Math.floor(s.mouseY) - erase_size);
              y < Math.min(canvas_size, Math.floor(s.mouseY) + erase_size);
              y++
            ) {
              if (s.dist(s.mouseX, s.mouseY, x, y) < erase_size) {
                s.mask[mask_selected_index].set(x, y, col);
              }
            }
          }
          s.mask[mask_selected_index].updatePixels();

          // p5.Graphics object should be re-created because of a bug related to updatePixels().
          for (var update_g = 0; update_g < max_colors; update_g++) {
            var new_g = s.createGraphics(canvas_size, canvas_size);
            new_g.image(s.mask[update_g], 0, 0);
            s.mask[update_g].remove();
            s.mask[update_g] = new_g;
          }
        }
      }
    };

    s.clear_canvas = function () {
      for (var i = 0; i < max_colors; i++) {
        s.mask[i].clear();
      }
      s.body = null;
    };


    s.clear_mask = function (idx){
      s.mask[idx].clear()
    }


    s.updateImage = function (url) {
      sf = 1;
      s.body = s.loadImage(url);
    };
  };
}

function OriginalNameSpace() {
  return function (s) {
    s.setup = function () {
      s.pixelDensity(1);
      s.createCanvas(canvas_size, canvas_size);
      s.body = null;
      s.cursor(s.HAND);
      s.yaw_angle = 0;
      s.r_x = Array(max_colors).fill(0);
      s.r_y = Array(max_colors).fill(0);
      s.d_x = Array(max_colors).fill(0);
      s.d_y = Array(max_colors).fill(0);
      mousePressed_here = false;
      s.direction_mode = false;

      // if (window.location.pathname == '/single'){
      //   let download_button = s.createButton('export');
      //   download_button.id('download-button-original')
      //   download_button.position($('#defaultCanvas0').width()-60, 5);

      //   let left_button = s.createButton('<');
      //   left_button.id('left-button')
      //   left_button.addClass('styled-button lr-button')
      //   left_button.position(10, $('#defaultCanvas0').height()/2-10);

      //   let right_button = s.createButton('>');
      //   right_button.id('right-button')
      //   right_button.addClass('styled-button lr-button')
      //   right_button.position($('#defaultCanvas0').height()-30, $('#defaultCanvas0').height()/2-10);
      // }
    };

    s.draw = function () {
      s.background(255);
      s.noTint();
      if (s.body != null) {
        s.image(s.body, 0, 0, s.width, s.height);
      }
      s.tint(255, 127);

      for (var i = 0; i < max_colors; i++) {
        s.image(p5_input_reference.mask[i], s.r_x[i], s.r_y[i]);
      }
    };

    s.mousePressed = function (e) {
      s.d_x[mask_selected_index] = s.mouseX;
      s.d_y[mask_selected_index] = s.mouseY;

      if (
        s.mouseX <= s.width &&
        s.mouseX >= 0 &&
        s.mouseY <= s.height &&
        s.mouseY >= 0
      ) {
        s.mousePressed_here = true;
      }
    };

    // s.keyPressed = function () {
    //   // keyboard press E
    //   if (s.keyCode === 69) {
    //     if (s.direction_mode) $("#p5-original").css({ outline: "none" });
    //     else $("#p5-original").css({ outline: "solid" });
    //     s.direction_mode = !s.direction_mode;
    //   }

    //   // keyboar left key, right key
    //   if (s.direction_mode & (s.keyCode === 37)) {
    //     s.yaw_angle -= 4;
    //     s.yaw_angle = Math.max(s.yaw_angle, -16);
    //     updateOrigin();
    //   }
    //   if (s.direction_mode & (s.keyCode === 39)) {
    //     s.yaw_angle += 4;
    //     s.yaw_angle = Math.min(s.yaw_angle, 16);
    //     updateOrigin();
    //   }
    // };

    s.mouseReleased = function (e) {
      s.mousePressed_here = false;
    };

    s.mouseDragged = function (e) {
      if (ui_uninitialized || s.mousePressed_here == false) return;
      if (s.direction_mode) {
      } else {
        if (
          s.mouseX <= s.width &&
          s.mouseX >= 0 &&
          s.mouseY <= s.height &&
          s.mouseY >= 0
        ) {
          s.r_x[mask_selected_index] += s.mouseX - s.d_x[mask_selected_index];
          s.r_y[mask_selected_index] += s.mouseY - s.d_y[mask_selected_index];

          s.d_x[mask_selected_index] = s.mouseX;
          s.d_y[mask_selected_index] = s.mouseY;
        }
      }
    };

    s.updateImage = function (url) {
      s.body = s.loadImage(url);
    };

    s.clear_canvas = function () {
      
      s.yaw_angle = 0;
      s.body = null;

      for (var i = 0; i < max_colors; i++) {
        s.r_x[i] = 0;
        s.r_y[i] = 0;
        s.d_x[i] = 0;
        s.d_y[i] = 0;
      }
    };
  };
}

function generateOutputNameSpace() {
  return function (s) {
    s.setup = function () {
      s.pixelDensity(1);
      let cvs = s.createCanvas(canvas_size, canvas_size);

      s.images = [];
      s.currentImage = 0;
      // let download_button = s.createButton('export');
      // download_button.id('download-button')
      // download_button.position($('#defaultCanvas2').width()-60, 5);
      // download_button.parent(cvs)
    };

    s.draw = function () {
      s.background(255);
      if (s.images.length > s.currentImage) {
        s.background(255);
        s.image(s.images[s.currentImage], 0, 0, s.width, s.height);
      }
    };

    s.updateImages = function (urls) {
      for (var i = urls.length - 1; i >= 0; i--) {
        var img = s.loadImage(urls[i]);
        s.images[i] = img;
      }
      s.currentImage = urls.length - 1;
    };

    s.changeCurrentImage = function (index) {
      if (index < s.images.length) {
        s.currentImage = index;
      }
    };

    s.clear_canvas = function () {
      s.images = [];
      s.currentImage = 0;
    };
  };
}

function MaskNameSpace(idx) {
  return function (s) {
    s.setup = function () {
      s.pixelDensity(1);
      s.createCanvas(mask_thumnail_size, mask_thumnail_size);
      s.background("white");

      s.body = null;
      s.idx = idx;
    };

    s.draw = function () {
      s.clear();
      s.background("white");
      s.body = p5_input_reference.mask[s.idx];
      s.image(s.body, 0, 0, mask_thumnail_size, mask_thumnail_size);
    };
  };
}

function add_new_mask(idx) {
  $("#palette-body").append(
    `<div class="card md-3 palette-item palette-item-class" 
      id="palette-${idx}"
      style="background-color: ${colors[idx]};
      border: dashed ${colors[idx]}"></div>`
  );

  $("#palette-" + idx).append(
    `<div class='card-img' id = "${"#palette-" + idx + "-mask"}" />`
  );
  mask_thumnail = new p5(MaskNameSpace(idx), "#palette-" + idx + "-mask");
  $(".palette-item.selected").removeClass("selected");
  $("#palette-" + idx).click(function () {
    $(".palette-item.selected").removeClass("selected");
    $(this).addClass("selected");
    mask_selected_index = idx;
  });
  $("#palette-" + idx).click();
}

function updateOriginRandomGenerate() {
  let gender = $("input[name='gender']:checked").val();
  let hair = $("input[name='hair']:checked").val();
  let eye = $("input[name='eye']:checked").val();

  $.ajax({
    type: "POST",
    url: "/post",
    data: JSON.stringify({
      id: id,
      seed_key : `${gender}_${hair}_${eye}`,
      type: "random_generate"
    }),
    dataType: "json",
    contentType: "application/json",
  }).done(function (data, textStatus, jqXHR) {
    let paths = data["result"];
  
    $('.random-sample').each((i, d)=>{
      $(d).attr('src', paths[i]);
    })
    // url = $('.random-sample.selected').attr('src');
    // p5_input_original.updateImage(url);
    // original_choose = url;

    p5_input_reference.clear_canvas();
    $(".palette-item-class").remove();
    
    mask_selected_index = null;
    mask_idx = 0;
    mask_selected_index = 0;
    selected_class =  $('.random-sample.selected').attr('src');
    
    palette = selected_class;
    add_new_mask(mask_idx);

    p5_input_reference.updateImage(selected_class);

    updateOriginRandomGenerateNoise()

    

  });
}

function updateOriginRandomGenerateNoise() {
  
  random_seed = parseInt($('.random-sample.selected').attr('src').split('/').slice(-1)[0].split('.')[0]);
  
  $.ajax({
    type: "POST",
    url: "/post",
    data: JSON.stringify({
      id: id,
      random_seed:random_seed,
      type: "random_generate_noise"
    }),
    dataType: "json",
    contentType: "application/json",
  }).done(function (data, textStatus, jqXHR) {
    let paths = data["result"];
    
    $('.random-sample-noise').each((i, d)=>{
      $(d).attr('src', paths[i]);
    })
    
    for(let i = 0 ; i < 5 ; i++){
      $(`#sefa${i}`).slider("refresh");
    }
    
    p5_input_reference.clear_canvas();
    $(".palette-item-class").remove();
    pose_lr = 0;
    mask_selected_index = null;
    mask_idx = 0;
    mask_selected_index = 0;
    selected_class =  $('.random-sample-noise.selected').attr('src');
    
    palette = selected_class;
    add_new_mask(mask_idx);

    p5_input_reference.updateImage(selected_class);
    updateDirectManipulateExample()
  });
}

function updateDirectManipulateExample() {
  let random_seed = selected_class.split('/').slice(-1)[0];
  // random_seed = parseInt($('.random-sample.selected').attr('src').split('/').slice(-1)[0].split('.')[0]);
  
  $.ajax({
    type: "POST",
    url: "/post",
    data: JSON.stringify({
      id: id,
      random_seed:random_seed,
      type: "direct_manipulation_example"
    }),
    dataType: "json",
    contentType: "application/json",
  }).done(function (data, textStatus, jqXHR) {
    let paths = data["result"];
    console.log(paths)
    $('.sefa-image').each((i, d)=>{
      $(d).attr('src', paths[i]);
    })
   
  });
}


function updateDirectManipulate() {
  let random_seed = selected_class.split('/').slice(-1)[0];
  $.ajax({
    type: "POST",
    url: "/post",
    data: JSON.stringify({
      id: id,
      random_seed:random_seed,
      manipulation : [
        parseInt($('#sefa0').val()),
        parseInt($('#sefa1').val()),
        parseInt($('#sefa2').val()),
        parseInt($('#sefa3').val()),
        parseInt($('#sefa4').val()),
      ],
      type: "direct_manipulation"
    }),
    dataType: "json",
    contentType: "application/json",
  }).done(function (data, textStatus, jqXHR) {
    let url = data['result']
    
    p5_input_reference.clear_canvas();
    $(".palette-item-class").remove();
    pose_lr = 0;
    mask_selected_index = null;
    mask_idx = 0;
    mask_selected_index = 0;
    // selected_class = url;
    p5_input_reference.updateImage(url);
    
    palette = url;
    add_new_mask(mask_idx);

    
  });
}

function waiting_start(){
  $('.waiting-modal').modal('show');  
}

function waiting_end(){
  $('.waiting-modal').modal('hide');
}

function updateResult() {
  disableUI();
  // waiting_start();
  var canvas_reference = $("#p5-reference canvas").slice(1);

  var data_reference = [];

  for (var canvas_i = 0; canvas_i < max_colors; canvas_i++) {
    data_reference.push(
      canvas_reference[canvas_i]
        .toDataURL("image/png")
        .replace(/data:image\/png;base64,/, "")
    );
  }
  var palettes = [];

  for (var canvas_i = 0; canvas_i < max_colors; canvas_i++) {
    palettes.push(palette);
  }

  $.ajax({
    type: "POST",
    url: "/post",
    data: JSON.stringify({
      type: "generate",
      id: id,
      original: original_choose,
      references: palettes,
      data_reference: data_reference,
      shift_original: [p5_input_original.r_x, p5_input_original.r_y],
      colors: colors,
      flip: sf
    }),
    dataType: "json",
    contentType: "application/json",
  }).done(function (data, textStatus, jqXHR) {
    let urls = data["result"];
    $("#ex1").slider({ max: urls.length - 1, setValue: urls.length - 1 });
    p5_output.updateImages(urls);
    $("#ex1").attr("data-slider-value", urls.length - 1);
    $("#ex1").slider("refresh");
    enableUI();
    // waiting_end();
  });
}

function enableUI() {
  ui_uninitialized = false;
  $("button").removeAttr("disabled");
  $("#ex1").slider("enable");
}

function disableUI() {
  ui_uninitialized = true;
  $("button").attr("disabled", true);
  $("#ex1").slider("disable");
}

$(function () {

  

  $("#main-ui-submit").click(function () {
    updateResult();
  });

  $("#sketch-clear").click(function () {
    p5_input_reference.clear_canvas();
    p5_input_original.clear_canvas();
    p5_output.clear_canvas();
    $(".palette-item-class").remove();
    palette = null;
    original_image = null;
    original_choose = null;
    mask_selected_index = null;
    mask_idx = 0;
    mask_selected_index = 0;
    $("#palette-eraser").click();

    $("#sketch-clear").attr("disabled", true);
    $("#main-ui-submit").attr("disabled", true);
  });

  $("#mask-clear").click(function () {
    p5_input_reference.clear_mask(mask_selected_index);
  });

  for (let idx = 0; idx < image_paths.length; idx++) {
    let [dir_path, dirs, files] = image_paths[idx];
    let dir_name = dir_path.split("/").reverse()[0];

    if (dir_name === "") dir_name = "etc";
    if (dir_name=='etc'){
      continue;
    }

    
    $("#class-picker").append(`<optgroup id="${dir_name}" label="${String(webtoon_id[dir_name])}">`);
    $("#upload-select").append(
      `<option value="${dir_name}">${webtoon_id[dir_name]}</option>>`
    );

    for (var i = 0; i < files.length; i++) {
      
      var image_name = files[i];
      let image_path = (dir_path == "etc" ? "" : dir_path + "/") + image_name;
      $(`#${dir_name}`).append(
        '<option data-img-src="' +
          image_path +
          '" data-img-alt="' +
          image_name +
          '" value="' +
          image_name +
          '">' +
          image_name +
          "</option>"
      );
    }
    $("#class-picker").append("</optgroup>");
  }
  $("#class-picker").imagepicker({
    hide_select: true,
  });


  $("#class-picker-submit-reference").after(
    '<div class="row" id="class-picker-ui"></div>'
  );
  $("#class-picker").appendTo("#class-picker-ui");
  $("#class-picker-submit-original").appendTo("#class-picker-ui");
  $("#class-picker-submit-reference").appendTo("#class-picker-ui");

  

  mask_idx = 0;

  $("#class-picker-submit-reference").click(function () {
    p5_input_reference.clear_canvas();
    $(".palette-item-class").remove();
    mask_selected_index = null;
    mask_idx = 0;
    mask_selected_index = 0;
    selected_class = $("#class-picker option:selected").attr("data-img-src");
    
    palette = selected_class;
    add_new_mask(mask_idx);

    p5_input_reference.updateImage(selected_class);
    enableUI();
  });

  $(document).on('dblclick', 'img.image_picker_image',function(){
    p5_input_reference.clear_canvas();
    $(".palette-item-class").remove();
    pose_lr = 0;
    mask_selected_index = null;
    mask_idx = 0;
    mask_selected_index = 0;
    selected_class = $("#class-picker option:selected").attr("data-img-src");
    
    palette = selected_class;
    add_new_mask(mask_idx);

    p5_input_reference.updateImage(selected_class);
    enableUI();
  });



  $("#add-mask").click(function () {
    mask_idx += 1;
    add_new_mask(mask_idx);
  });

  $("#class-picker-submit-original").click(function () {
    // selected_class = $("#class-picker option:selected").attr("data-img-src");
    // selected_class = 
    p5_input_original.updateImage(selected_class);
    original_image = selected_class;
    original_choose = selected_class;
    
    enableUI();
  });

  $("#palette-eraser").click(function () {
    $(".palette-item.selected").removeClass("selected");
    $(this).addClass("selected");
  });
  
  
  
  $('#flip').click(function(){
    sf *= -1;
    console.log(sf)
  })
  
  
  $("#ex0").slider();
  $("#ex1").slider({
    formatter: function (value) {
      return "interpolation: " + (value / (16 - 1)).toFixed(2);
    },
  });
  $("#ex1").slider("disable");
  $("#ex1").change(function () {
    p5_output.changeCurrentImage(parseInt($("#ex1").val()));
  });

  
  
  

  p5_input_reference = new p5(ReferenceNameSpace(), "p5-reference");
  p5_input_original = new p5(OriginalNameSpace(), "p5-original");
  p5_output = new p5(generateOutputNameSpace(), "p5-right");

  // Image Upload
  var $uploadCrop, tempFilename, rawImg, imageId, file_name;

  function readFile(input) {
    if (input.files && input.files[0]) {
      var reader = new FileReader();
      reader.onload = function (e) {
        $(".upload-demo").addClass("ready");
        $("#cropImagePop").modal("show");
        rawImg = e.target.result;
      };
      reader.readAsDataURL(input.files[0]);
      file_name = input.files[0].name;
    } else {
      swal("Sorry - you're browser doesn't support the FileReader API");
    }
  }

  $uploadCrop = $("#upload-demo").croppie({
    viewport: {
      width: 256,
      height: 256,
    },
    enforceBoundary: false,
    enableExif: true,
  });

  $("#cropImagePop").on("shown.bs.modal", function () {
    $uploadCrop
      .croppie("bind", {
        url: rawImg,
      })
      .then(function () {
        console.log("jQuery bind complete");
      });
  });

  $(".item-img").on("change", function () {
    imageId = $(this).data("id");
    tempFilename = $(this).val();
    $("#cancelCropBtn").data("id", imageId);
    readFile(this);
  });

  $("#cropImageBtn").on("click", function (ev) {
    // waiting_start()
    $uploadCrop
      .croppie("result", {
        type: "base64",
        format: "jpeg",
        size: { width: 256, height: 256 },
      })
      .then(function (resp) {
        let directory = $("#upload-select option:selected").val();
        $("#cropImagePop").modal("hide");

        $.ajax({
          type: "POST",
          url: "/post",
          data: JSON.stringify({
            type: "upload",
            id: id,
            image: resp,
            directory: directory,
            file_name: file_name,
          }),
          dataType: "json",
          contentType: "application/json",
        }).done(function (data, textStatus, jqXHR) {
          let url = data["result"];
          console.log(url)
          selected_class2 = url;
          p5_input_original.updateImage(selected_class2);
          original_image = selected_class2;
          original_choose = selected_class2;
          
          enableUI();
          $('input[type="file"]').val(null);
          console.log('modal end')
          // waiting_end()
          // $('.waiting-modal').modal('hide');
          // window.location.href = window.location.href;
        });
      });
  });

  // Image Download
  $(document).on('click', '#download-button', function () {
    
    var canvas = $("#p5-right canvas").get()[0];
    image = canvas
      .toDataURL("image/png")
      .replace("image/png", "image/octet-stream");
    var link = document.createElement("a");
    link.download = "edited.png";
    link.href = image;
    link.click();
  });

  // Image Download
  $(document).on('click', '#download-button-original', function () {
    
  var canvas = $("#p5-original canvas").get()[0];
  image = canvas
    .toDataURL("image/png")
    .replace("image/png", "image/octet-stream");
  var link = document.createElement("a");
  link.download = "edited.png";
  link.href = image;
  link.click();
});

  $(function () {
    $('[data-toggle="popover"]').popover();
  });

  // image select
  $(document).on('click', '.random-sample',function(){
    
    $('.random-sample.selected').removeClass('selected');
    $(this).addClass('selected');
  });

  $(document).on('dblclick', '.random-sample',function(){
    for(let i = 0 ; i < 5 ; i++){
      $(`#sefa${i}`).slider("refresh");
    }
    updateOriginRandomGenerateNoise()
    
  });

  $(document).on('dblclick', '.random-sample-noise',function(){
    for(let i = 0 ; i < 5 ; i++){
      $(`#sefa${i}`).slider("refresh");
    }
    p5_input_reference.clear_canvas();
    $(".palette-item-class").remove();
    pose_lr = 0;
    mask_selected_index = null;
    mask_idx = 0;
    mask_selected_index = 0;
    selected_class =  $('.random-sample-noise.selected').attr('src');
    
    palette = selected_class;
    add_new_mask(mask_idx);

    p5_input_reference.updateImage(selected_class);
    updateDirectManipulateExample()
    
  });

  $(document).on('click', '.random-sample-noise',function(){
    
    $('.random-sample-noise.selected').removeClass('selected');
    $(this).addClass('selected');
  });

  // $(document).on('click', '#right-button',function(){
  //   pose_lr += 3;
  //   if (pose_lr === 15){
  //     $('#right-button').attr('disabled', true);
  //   }
  //   if (pose_lr > -15){
  //     $('#left-button').attr('disabled', false);
  //   }
  //   updateOriginRandom()
  // });


  // $(document).on('click', '#left-button',function(){
    
  //   pose_lr -= 3;
  //   if (pose_lr === 15){
  //     $('#left-button').attr('disabled', true);
  //   }
  //   if (pose_lr < 15){
  //     $('#right-button').attr('disabled', false);
  //   }
  //   updateOriginRandom()
  // });

  $('.again').click(()=>{
    updateOriginRandomGenerate()
    
  })

  $('.submit').click(()=>{

    updateOriginRandomGenerate()
    
  })

  $('.again-noise').click(()=>{
    // $('#randomSampleGeneratedNoise').empty()
    updateOriginRandomGenerateNoise()
  })

  for(let i = 0 ; i < 5 ; i++){
    $(`#sefa${i}`).slider()
    $(`#sefa${i}`).change(()=>updateDirectManipulate())
  } 

  

  // $( document ).ajaxStart(function() {
    
  //   // $('html').css("cursor", "wait"); 
  //   $('.waiting-modal').modal('show');
    
  // });
  // //AJAX ?????? ??????
  // $( document ).ajaxComplete(function() {
  //   $('.waiting-modal').modal('hide');
  //     // $('html').css("cursor", "auto"); 
      
  // });
  let output = $('#temp-save-box')
  $('.temp-save').click(function() {
    let imageClone = $('.random-sample.selected').clone().attr('class', 'temp-save-img temp-save-img-generated');
    output.append(imageClone);
    
  });

  $('.temp-save-noise').click(function() {
    let imageClone = $('.random-sample-noise.selected').clone().attr('class', 'temp-save-img temp-save-img-generated');
    output.append(imageClone);
    
  });

  $('.temp-save-reference').click(function() {
    let imageClone = $('.thumbnail.selected .image_picker_image').clone().attr('class', 'temp-save-img temp-save-img-reference');
    output.append(imageClone);
    
  });  

  function takeshot() {
    let div = document.getElementById('detail-comment'); // $('#detail-comment');

    html2canvas(div).then(
        function (canvas) {
          let image = canvas
          .toDataURL("image/png")
          .replace("image/png", "image/octet-stream");
            let link = document.createElement("a");
            link.download = "comment.png";
            link.href = image;
            link.click();

        })
  }
  $('#comment-export').click(()=>{

    takeshot()
    
  })
  
  $(document).on('click', '.temp-save-img',function(){
    if($(this).hasClass('selected')){
      $(this).removeClass('selected')
    }
    else{
      $(this).addClass('selected');
    }
  });

  $('#temp-save-img-delete').on('click', function() {
    $('.temp-save-img.selected').remove();
    
  });

  $(document).on('dblclick', '.temp-save-img',function(){
    for(let i = 0 ; i < 5 ; i++){
      $(`#sefa${i}`).slider("refresh");
    }
    p5_input_reference.clear_canvas();
    $(".palette-item-class").remove();
    pose_lr = 0;
    mask_selected_index = null;
    mask_idx = 0;
    mask_selected_index = 0;
    selected_class =  $(this).attr('src');
    
    palette = selected_class;
    add_new_mask(mask_idx);

    p5_input_reference.updateImage(selected_class);
    updateDirectManipulateExample()
  });
});


