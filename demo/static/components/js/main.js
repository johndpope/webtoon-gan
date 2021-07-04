/*
WebotoonGAN
Copyright (c) 2021-present Hyunkwon Ko, Subin An

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
*/

// Refer to https://github.com/quolc/neural-collage/blob/master/static/demo_feature_blending/js/main.js

max_colors = 9;
colors = [
  "#FE2712",
  "#66B032",
  "#FEFE33",
  "#FE2712",
  "#66B032",
  "#FEFE33",
  "#FE2712",
  "#66B032",
  "#FEFE33",
];
original_image = null;
original_choose = null;

palette = [];
mask_selected_index = 0;

ui_uninitialized = true;
p5_input_original = null;
p5_input_reference = null;
p5_output = null;
sync_flag = true;
id = null;
brush_size = 20;
sf = 1;
mask_thumnail_size = 75;

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
      s.rotate(s.rotate_angle);
      // s.scale(sf);
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
      // s.scale(sf);
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
      // keyboar r
      if (s.keyCode === 82) {
        if (s.rotate_mode) $("#p5-reference").css({ outline: "none" });
        else $("#p5-reference").css({ outline: "solid" });
        s.rotate_mode = !s.rotate_mode;
      }

      // keyboard <>
      if (s.rotate_mode & (s.keyCode === 188)) {
        s.rotate_angle -= s.PI / 8;
      }

      if (s.rotate_mode & (s.keyCode === 190)) {
        s.rotate_angle += s.PI / 8;
      }

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

    s.clear_mask = function (idx) {
      s.mask[idx].clear();
    };

    s.updateImage = function (url) {
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

    s.keyPressed = function () {
      // keyboard press E
      if (s.keyCode === 69) {
        if (s.direction_mode) $("#p5-original").css({ outline: "none" });
        else $("#p5-original").css({ outline: "solid" });
        s.direction_mode = !s.direction_mode;
      }

      // keyboar left key, right key
      if (s.direction_mode & (s.keyCode === 37)) {
        s.yaw_angle -= 4;
        s.yaw_angle = Math.max(s.yaw_angle, -16);
        updateOrigin();
      }
      if (s.direction_mode & (s.keyCode === 39)) {
        s.yaw_angle += 4;
        s.yaw_angle = Math.min(s.yaw_angle, 16);
        updateOrigin();
      }
    };

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
      $("#pose_lr").slider("refresh");
      // $("#pose_lr").slider("disable");
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
      s.createCanvas(canvas_size, canvas_size);

      s.images = [];
      s.currentImage = 0;
      s.frameRate(15);
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

function updateOrigin() {
  $.ajax({
    type: "POST",
    url: "/post",
    data: JSON.stringify({
      type: "original",
      id: id,
      original: original_image,
      distance: [-parseInt($("#pose_lr").val()), 0],
    }),
    dataType: "json",
    contentType: "application/json",
  }).done(function (data, textStatus, jqXHR) {
    let url = data["result"];
    p5_input_original.updateImage(url);
    original_choose = url;
  });
}

function updateResult() {
  disableUI();
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
    palettes.push(palette[0]);
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
    palette = [];
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
    $("#class-picker").append(`<optgroup id="${dir_name}" label=${dir_name}>`);
    $("#upload-select").append(
      `<option value="${dir_name}">${dir_name}</option>>`
    );

    for (var i = 0; i < files.length; i++) {
      console.log(files[i]);
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
    hide_select: false,
  });

  $("#class-picker").after(
    '<button type="submit" class="form-control btn btn-success col-md-2" id="class-picker-submit-original">add to original</button>'
  );

  $("#class-picker").after(
    '<button type="submit" class="form-control btn btn-primary col-md-2" id="class-picker-submit-reference">add to reference</button>'
  );

  $("#class-picker-submit-reference").after(
    '<div class="row" id="class-picker-ui"></div>'
  );
  $("#class-picker").appendTo("#class-picker-ui");
  $("#class-picker-submit-original").appendTo("#class-picker-ui");
  $("#class-picker-submit-reference").appendTo("#class-picker-ui");

  function add_new_mask(idx) {
    $("#palette-body").append(
      '<div class="card md-3 palette-item palette-item-class" id="palette-' +
        idx +
        '"' +
        '" style="background-color: ' +
        colors[idx] +
        ';"></li>'
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

  mask_idx = 0;

  $("#class-picker-submit-reference").click(function () {
    p5_input_reference.clear_canvas();
    $(".palette-item-class").remove();
    mask_selected_index = null;
    mask_idx = 0;
    mask_selected_index = 0;
    selected_class = $("#class-picker option:selected").attr("data-img-src");
    palette.push(selected_class);
    add_new_mask(mask_idx);

    p5_input_reference.updateImage(selected_class);
    enableUI();
  });

  $("#add-mask").click(function () {
    mask_idx += 1;
    add_new_mask(mask_idx);
  });

  $("#class-picker-submit-original").click(function () {
    selected_class = $("#class-picker option:selected").attr("data-img-src");

    p5_input_original.updateImage(selected_class);
    original_image = selected_class;
    original_choose = selected_class;
    $("#pose_lr").slider("refresh");
    enableUI();
  });

  $("#palette-eraser").click(function () {
    $(".palette-item.selected").removeClass("selected");
    $(this).addClass("selected");
  });
  $("#pose_lr").slider();
  // $("#pose_lr").slider("disable");
  $("#pose_lr").change(() => updateOrigin());

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
          window.location.href = window.location.href;
        });
      });
  });

  // Image Download

  $("#download-button").on("click", function () {
    var canvas = $("#p5-right canvas").get()[0];
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
});

$("#myModal").on("shown.bs.modal", function () {
  $("#myInput").trigger("focus");
});
