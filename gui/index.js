window.onload = function () {

    var rangeSlider = function () {

        var slider = $('.range-slider'),
            range = $('.range-slider__range'),
            value = $('.range-slider__value');

        slider.each(function () {

            value.each(function () {
                var value = $(this).prev().attr('value');
                $(this).html(value);
            });

            range.on('input', function () {
                var id = this.getAttribute('data-id');
                if (id) {
                    document.getElementById(id).setAttribute('data-style', this.value);
                } else {
                    this.setAttribute('image-id', this.value);
                }
                $(this).next(value).html(this.value);
                render_button();
            });
        });
    };

    rangeSlider();

    var container = document.getElementById("resize-container");
    var container_rect = interact.getElementRect(container);
    var containerOffsetLeft = container_rect.left;
    var containerOffsetTop = container_rect.top;
    var nuclei_to_color_map = {'neutrophil': 'black', 'epithelial': 'green', 'lymphocyte': 'orange', 'plasma': 'red', 'eosinophil': 'blue', 'connectivetissue': 'fuchsia'};

    console.log("Here your attention");
    console.log(containerOffsetLeft);
    console.log(containerOffsetTop);

    function dragMoveListener(event) {
        console.log('dragMove');

        var target = event.target,

        // keep the dragged position in the data-x/data-y attributes
        x = (parseFloat(target.getAttribute('data-x')) || 0) + event.dx,
        y = (parseFloat(target.getAttribute('data-y')) || 0) + event.dy;

        // translate the element
        target.style.webkitTransform =
            target.style.transform =
                'translate(' + (x-55) + 'px, ' + (y-37) + 'px)';

        // update the position attributes
        target.setAttribute('data-x', x);
        target.setAttribute('data-y', y);
        selectItem(event, event.target);
        // render_button();
    }

// this is used later in the resizing and gesture demos
    window.dragMoveListener = dragMoveListener;

    interact('.resize-drag')
        .draggable({
            onmove: window.dragMoveListener,
            onend: render_button,
            restrict: {
                restriction: 'parent',
                elementRect: {top: 0, left: 0, bottom: 1, right: 1},
            },
        })
        .on('tap', function (event) {
            console.log('tap');
            var target = event.target;
            var size = parseInt(target.getAttribute('data-size'));
            var new_size = (size + 1) % 20;
            target.setAttribute('data-size', new_size);
            target.style.fontSize = sizeToFont(new_size);
            // $(event.currentTarget).remove();
            selectItem(event, event.target);
            render_button();
            event.preventDefault();
        })
        .on('hold', function (event) {
            console.log('hold');
            $(event.currentTarget).remove();
            render_button();
            event.preventDefault();
        });

    function selectItem(event, target, should_deselect) {
        event.stopPropagation();
        var hasClass = $(target).hasClass('selected');
        $(".resize-drag").removeClass("selected");
        $('#range-slider').attr('data-id', '');
        if (should_deselect && hasClass) {
        } else {
            $(target).addClass("selected");
            $('#range-slider').attr('data-id', target.id);
            var style = target.getAttribute('data-style');
            style = style ? style : -1;
            $('#range-slider').val(style);
            $('.range-slider__value').text(style.toString());
        }
    }

    $(".resize-drag").click(function (e) {
        $(".resize-drag").removeClass("selected");
        $(this).addClass("selected");
        e.stopPropagation();
    });

    function guidGenerator() {
        var S4 = function () {
            return (((1 + Math.random()) * 0x10000) | 0).toString(16).substring(1);
        };
        return (S4() + S4() + "-" + S4() + "-" + S4() + "-" + S4() + "-" + S4() + S4() + S4());
    }

    function stuff_add(evt) {
        evt.stopPropagation();

        var newContent = document.createTextNode('*');
        var node = document.createElement("DIV");
        node.className = "resize-drag";
        node.id = guidGenerator();
        node.appendChild(newContent);
        var init_size = 6;
        node.setAttribute('data-size', init_size);
        node.style.fontSize = sizeToFont(init_size);
        node.style.color = nuclei_to_color_map[evt.currentTarget.textContent.toLowerCase()]
        document.getElementById("resize-container").appendChild(node);

        // translate the element
        var rect = interact.getElementRect(node);
        var left = (rect.left - containerOffsetLeft)
        var top = (rect.top - containerOffsetTop)
        x_relative_shift = -left
        y_relative_shift = -top
        node.style.transform = 'translate(' + x_relative_shift + 'px, ' + y_relative_shift + 'px)';
        node.setAttribute('data-x', x_relative_shift);
        node.setAttribute('data-y', y_relative_shift);

        render_button();
    }

    function sizeToFont(size) {
        return size * 8 + 20;
    }

    document.querySelectorAll("ul.drop-menu > li").forEach(function (e) {
        e.addEventListener("click", stuff_add)
    });
    $(window).click(function (devt) {
        if (!devt.target.getAttribute('data-size') && !devt.target.getAttribute('max')) {
            $(".resize-drag").removeClass("selected");
            var image_style = $('#range-slider').attr('image-id');
            image_style = image_style ? parseInt(image_style) : -1;
            $('#range-slider').val(image_style);
            $('.range-slider__value').text(image_style.toString());
            $('#range-slider').attr('data-id', '');
        }
    });
};

display_benign = false
display_malignant = false
display_benign_addchild = false
display_malignant_addchild = false

function sizeToFont(size) {
        return size * 8 + 20;
    }

function guidGenerator() {
        var S4 = function () {
            return (((1 + Math.random()) * 0x10000) | 0).toString(16).substring(1);
        };
        return (S4() + S4() + "-" + S4() + "-" + S4() + "-" + S4() + "-" + S4() + S4() + S4());
    }

function refresh_image(response) {
        response = JSON.parse(response);
        document.getElementById("img_pred").src = response.img_pred;
        document.getElementById("layout_pred").src = response.layout_pred;
        document.getElementById("graph_pred").src = response.graph_pred;
    }

function addRow(obj, size, location, feature) {
        return;
        // Get a reference to the table
        let tableRef = document.getElementById('table').getElementsByTagName('tbody')[0];

        // Insert a row at the end of the table
        let newRow = tableRef.insertRow(-1);

        // Insert a cell in the row at index 0
        newRow.insertCell(0).appendChild(document.createTextNode(obj));
        newRow.insertCell(1).appendChild(document.createTextNode(size + ''));
        newRow.insertCell(2).appendChild(document.createTextNode(location + ''));
        newRow.insertCell(3).appendChild(document.createTextNode(feature + ''));
    }

function drawBenignTissueImage()
{
    display_benign = true
    display_malignant = false

    display_benign_addchild = true
    display_malignant_addchild = false

    render_button()
}

function drawMalignantTissueImage()
{
    display_malignant = true
    display_benign = false

    display_benign_addchild = false
    display_malignant_addchild = true

    render_button()
}

function render_button() {


    var allObjects = [];
    $("tbody").children().remove();

    var container = document.getElementById("resize-container");
    var container_rect = interact.getElementRect(container);
    var containerOffsetLeft = container_rect.left;
    var containerOffsetTop = container_rect.top;
    var containerWidth = container_rect.width;
    var containerHeight = container_rect.height;

    var nuclei_to_color_map = {'neutrophil': 'black', 'epithelial': 'green', 'lymphocyte': 'orange', 'plasma': 'red', 'eosinophil': 'blue', 'connectivetissue': 'fuchsia'};
    var color_to_nuclei_map = {'black': 'neutrophil', 'green': 'epithelial', 'orange': 'lymphocyte', 'red': 'plasma', 'blue': 'eosinophil', 'fuchsia': 'connectivetissue'};

    var feature_value = "malignant" //benign tissue
    var points = []
    var object_names = []

    if(display_benign || display_malignant){

        if(display_benign)
        {
            feature_value = "benign" //benign tissue
            points = [[74, 0], [17, 0], [56, 0], [30, 9], [199, 16], [124, 15], [47, 24], [30, 25], [17, 51], [23, 64], [27, 98], [2, 99], [0, 107], [23, 136], [40, 144], [194, 144], [198, 154], [56, 162], [13, 164], [233, 164], [24, 166], [0, 181], [59, 182], [43, 186], [195, 191], [86, 193], [29, 193], [62, 200], [40, 208], [108, 210], [88, 211], [114, 214], [138, 228], [194, 231], [98, 235], [90, 236], [110, 241], [125, 243], [163, 244], [230, 221], [186, 163], [176, 159], [177, 173], [164, 173], [153, 175], [138, 173], [121, 173], [110, 170], [106, 164], [98, 159], [90, 154], [80, 151], [78, 142], [70, 131], [68, 122], [60, 101], [59, 89], [61, 73], [60, 60], [65, 44], [76, 29], [96, 18], [109, 14], [139, 17], [154, 20], [163, 29], [165, 21], [174, 25], [187, 31], [197, 42], [206, 52], [209, 66], [216, 90], [211, 102], [213, 110], [207, 134], [9, 21], [12, 83], [247, 161], [242, 175], [226, 195], [225, 209], [3, 15], [15, 33], [14, 90], [229, 235], [239, 184], [230, 247], [248, 203], [92, 0], [209, 0], [190, 0], [43, 3], [66, 4], [211, 4], [84, 9], [0, 9], [220, 11], [44, 12], [61, 24], [49, 48], [37, 51], [40, 95], [36, 133], [0, 143], [16, 145], [5, 153], [0, 157], [39, 158], [29, 171], [203, 171], [44, 171], [55, 188], [204, 201], [155, 215], [97, 215], [152, 223], [100, 225], [165, 227], [164, 235], [153, 236], [197, 241], [72, 243], [102, 248], [68, 254], [152, 245], [46, 227], [14, 130], [76, 200], [98, 203], [140, 214], [61, 219]]
            object_names = ['lymphocyte', 'plasma', 'lymphocyte', 'lymphocyte', 'lymphocyte', 'epithelial', 'lymphocyte', 'plasma', 'epithelial', 'epithelial', 'lymphocyte', 'epithelial', 'epithelial', 'lymphocyte', 'lymphocyte', 'epithelial', 'epithelial', 'plasma', 'lymphocyte', 'plasma', 'lymphocyte', 'epithelial', 'lymphocyte', 'lymphocyte', 'plasma', 'lymphocyte', 'epithelial', 'plasma', 'epithelial', 'lymphocyte', 'lymphocyte', 'lymphocyte', 'lymphocyte', 'lymphocyte', 'plasma', 'plasma', 'plasma', 'lymphocyte', 'lymphocyte', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'lymphocyte', 'lymphocyte', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'epithelial', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue'];
        }

        if(display_malignant)
        {
            feature_value = "malignant" //malignant tissue
            points = [[47, 169], [146, 4], [158, 8], [173, 22], [185, 26], [243, 32], [180, 48], [205, 58], [1, 82], [11, 90], [51, 97], [37, 99], [79, 122], [201, 131], [155, 131], [252, 131], [203, 142], [153, 145], [145, 148], [71, 166], [225, 190], [82, 216], [21, 249], [98, 146], [94, 141], [87, 133], [79, 104], [85, 101], [102, 106], [113, 109], [52, 138], [144, 186], [145, 196], [143, 204], [154, 208], [145, 223], [158, 219], [170, 226], [175, 235], [81, 233], [77, 223], [70, 227], [63, 232], [48, 233], [37, 233], [28, 235], [41, 13], [28, 11], [32, 0], [21, 0], [217, 41], [216, 53], [227, 46], [226, 60], [132, 227], [55, 201], [60, 211], [45, 196], [43, 186], [120, 103], [129, 102], [65, 46], [119, 21], [119, 11], [132, 9], [201, 0], [211, 155], [199, 154], [185, 163], [184, 152], [177, 155], [173, 140], [164, 140], [165, 148], [122, 123], [128, 134], [138, 132], [141, 119], [143, 112], [215, 102], [208, 103], [198, 97], [229, 99], [234, 109], [244, 98], [192, 241], [202, 249], [209, 243], [225, 234], [232, 230], [254, 207], [210, 1], [233, 21], [96, 34], [237, 70], [244, 80], [9, 108], [52, 116], [10, 124], [0, 126], [186, 126], [172, 127], [20, 131], [0, 145], [113, 148], [117, 155], [31, 159], [15, 162], [131, 164], [0, 172], [241, 174], [129, 177], [9, 181], [97, 190], [113, 193], [0, 195], [100, 202], [81, 203], [124, 203], [111, 212], [0, 213], [244, 226], [0, 248], [218, 7], [237, 11], [97, 27], [97, 74], [110, 86], [18, 100], [48, 107], [4, 115], [17, 120], [49, 126], [18, 139], [108, 142], [7, 146], [0, 156], [43, 157], [128, 157], [4, 160], [136, 171], [232, 172], [11, 174], [1, 185], [124, 194], [0, 203], [247, 218], [0, 222], [15, 227], [6, 251], [62, 8], [64, 62], [245, 71], [79, 72], [106, 78], [236, 84], [114, 92], [182, 118], [160, 119], [176, 122], [105, 189], [75, 191], [93, 203], [130, 203], [113, 205], [221, 25], [229, 29], [111, 244], [1, 15], [16, 24], [29, 35], [0, 37], [20, 40], [33, 42], [8, 62], [137, 253], [52, 88], [61, 91], [60, 103], [223, 117], [37, 124], [36, 141], [230, 146], [29, 176], [32, 191], [49, 219], [26, 222], [192, 0], [195, 2], [104, 16], [70, 141], [58, 157]]
//            points = [[45, 175], [144, 8], [156, 14], [172, 31], [184, 35], [240, 35], [179, 53], [201, 64], [8, 94], [48, 101], [33, 101], [78, 126] , [199, 135], [153, 134], [248, 136], [199, 147], [149, 148], [142, 151], [67, 169], [220, 193], [79, 221], [19, 252], [96, 150] , [93, 145], [87, 138], [77, 110], [84, 107], [100, 112], [110, 116], [51, 145], [146, 191], [145, 200], [143, 208], [153, 212], [144, 227], [157, 224], [170, 230], [173, 239], [80, 238], [74, 229], [67, 232], [61, 238], [48, 240], [34, 240], [26, 243], [42, 19], [28, 17], [30, 4], [19, 4], [217, 47], [212, 56], [225, 52], [224, 65], [130, 234], [54, 206], [58, 215], [43, 200], [44 , 191], [119, 108], [130, 106], [68, 50], [117, 25], [118, 16], [131, 15], [199, 0], [210, 165], [196, 166], [182, 169], [180, 157], [172, 161], [170, 144], [160, 143], [161, 150], [121, 129], [125, 136], [136, 135], [142, 122], [141, 115], [212, 107], [204, 108], [195, 104], [229, 104], [231, 113], [241, 103], [192, 245], [201, 252], [210, 249], [222, 241], [230, 236], [251, 209], [209, 5], [231, 25], [97, 39], [233, 75], [243, 85], [8, 114], [49, 120], [7, 128], [185, 130], [170, 131], [17, 133], [111, 155], [118, 163], [31, 164], [11, 166], [129, 168], [240, 183], [251, 185], [128, 180], [6, 185], [93, 193], [109, 197], [97, 205] , [79, 207], [120, 208], [109, 215], [1, 219], [241, 230], [217, 12], [238, 17], [97, 31], [98, 78], [110, 89], [20, 107], [45, 111], [2, 120], [15, 125], [46, 130], [16, 144], [106, 151], [3, 150], [41, 160], [126, 162], [3, 166], [132, 173], [231, 177], [8, 177], [121, 198], [244, 222], [0, 227], [12, 232], [5, 253], [64, 15], [64, 66], [242, 76], [83, 74], [107, 82], [237, 89], [115, 94], [184, 123], [158, 124], [174, 126], [101, 192], [73, 196], [91, 208], [129, 209], [110, 208], [217, 28], [224, 31], [ 106, 246], [11, 26], [26, 38], [16, 43], [30, 44], [4, 66], [134, 254], [49, 91], [59, 95], [56, 107], [219, 121], [34, 128], [34, 145], [229, 151], [27, 179], [30, 198], [46, 222], [24, 225], [187, 0], [191, 5], [102, 19], [66, 147], [54, 159]]

            object_names = ['epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'lymphocyte', 'epithelial', 'epithelial', 'lymphocyte', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'lymphocyte', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'lymphocyte', 'lymphocyte', 'lymphocyte', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue']
//            object_names = ['epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'lymphocyte', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'lymphocyte', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'lymphocyte', 'lymphocyte', 'lymphocyte', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue']
        }

        for (var i = 0; i < points.length; i++) {

        var left = points[i][0]
        var top = points[i][1]

        var grid = 25 / 5;
        var text = object_names[i];
        var style = -1;
        var size = 0;

        allObjects.push({
            'left': left,
            'top': top,
            'text': text,
            'feature': feature_value
        });

        if (display_benign_addchild || display_malignant_addchild)
        {
                var newContent = document.createTextNode('*');
                var node = document.createElement("DIV");
                node.className = "resize-drag";
                node.id = guidGenerator();
                node.appendChild(newContent);
                node.style.color = nuclei_to_color_map[text]
                var init_size = 6;
                node.setAttribute('data-size', init_size);
                node.style.fontSize = sizeToFont(init_size);

                document.getElementById("resize-container").appendChild(node);

                // translate the element
                var rect = interact.getElementRect(node);
                var left_scaled = (left/255.0)*(containerWidth);
                var top_scaled = (top/255.0)*(containerHeight)
                x_relative_shift = left_scaled - (rect.left - containerOffsetLeft)// - 55
                y_relative_shift = top_scaled - (rect.top - containerOffsetTop)// - 37
                node.style.transform = 'translate(' + x_relative_shift + 'px, ' + y_relative_shift + 'px)';

                node.setAttribute('data-x', x_relative_shift);
                node.setAttribute('data-y', y_relative_shift);
        }

//        addRow(text, size, location, style);
      }

      display_benign_addchild = false
      display_malignant_addchild = false
    }

    var children = document.getElementsByClassName('resize-drag');

    if (children.length > points.length) {

        for (var i = points.length; i < children.length; i++) {

        var rect = interact.getElementRect(children[i]);
        var height = rect.height / containerHeight;
        var width = rect.width / containerWidth;
        var left = (rect.left - containerOffsetLeft) / containerWidth;
        var top = (rect.top - containerOffsetTop) / containerHeight;

        var sx0 = left;
        var sy0 = top;
        var sx1 = width + left;
        var sy1 = height + sy0;
        var mean_x_s = 0.5 * (sx0 + sx1);
        var mean_y_s = 0.5 * (sy0 + sy1);
        var grid = 25 / 5;
        var location = Math.round(mean_x_s * (grid - 1)) + grid * Math.round(mean_y_s * (grid - 1));
        var size = parseInt(children[i].getAttribute('data-size'));
        var text = color_to_nuclei_map[children[i].style.color];

        var style = children[i].getAttribute('data-style') ?
            parseInt(children[i].getAttribute('data-style')) :
            -1;

        if(children[i].style.color != "")
        {
          if(left>=0 && top>=0)
          {
             allObjects.push({
            'left': left,
            'top': top,
            'text': text,
            'feature': "normal"
            });
          }
        }

//        console.log(size, location, text);
//        addRow(text, size, location, style);
    }
    }

    //var image_id = document.getElementById('range-slider').getAttribute('image-id');
    var image_id = 0
    var image_feature = image_id ? parseInt(image_id) : -1;
    addRow('background', '-', '-', image_feature);

    var results = {'image_id': image_feature, 'objects': allObjects};

    var url = 'get_data?data=' + JSON.stringify(results);
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function () {
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
            refresh_image(xmlHttp.responseText);
    };
    xmlHttp.open("GET", url, true); // true for asynchronous
    xmlHttp.send(null);
}
