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
    var nuclei_to_color_map = {'neutrophil': 'black', 'epithelial': 'green', 'lymphocyte': 'yellow', 'plasma': 'red', 'eosinophil': 'blue', 'connectivetissue': 'fuchsia'};


    function dragMoveListener(event) {
        console.log('dragMove');
        var target = event.target,
            // keep the dragged position in the data-x/data-y attributes
            x = (parseFloat(target.getAttribute('data-x')) || 0) + event.dx,
            y = (parseFloat(target.getAttribute('data-y')) || 0) + event.dy;

        // translate the element
        target.style.webkitTransform =
            target.style.transform =
                'translate(' + x + 'px, ' + y + 'px)';

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
                elementRect: {top: 0, left: 0, bottom: 1, right: 1}
            },
        })
        .on('tap', function (event) {
            console.log('tap');
            var target = event.target;
            var size = parseInt(target.getAttribute('data-size'));
            var new_size = (size + 1) % 10;
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
        var init_size = 0;
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

display_511 = false
display_4427 = false

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
    display_511 = true
    display_4427 = false
    render_button()
}

function drawMalignantTissueImage()
{
    display_4427 = true
    display_511 = false
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

    var nuclei_to_color_map = {'neutrophil': 'black', 'epithelial': 'green', 'lymphocyte': 'yellow', 'plasma': 'red', 'eosinophil': 'blue', 'connectivetissue': 'fuchsia'};
    var color_to_nuclei_map = {'black': 'neutrophil', 'green': 'epithelial', 'yellow': 'lymphocyte', 'red': 'plasma', 'blue': 'eosinophil', 'fuchsia': 'connectivetissue'};

    var feature_style = "511" //benign tissue
    var points_511 = [[0.30078125, 0.0], [0.078125, 0.0078125], [0.234375, 0.0078125], [0.12890625, 0.05078125], [0.78515625, 0.0703125], [0.49609375, 0.0703125], [0.1953125, 0.10546875], [0.125, 0.11328125], [0.08203125, 0.22265625], [0.1015625, 0.2734375], [0.11328125, 0.3984375], [0.02734375, 0.40625], [0.0078125, 0.44140625], [0.09765625, 0.54296875], [0.171875, 0.58203125], [0.77734375, 0.58203125], [0.7890625, 0.6171875], [0.23046875, 0.640625], [0.0625, 0.65625], [0.91796875, 0.6484375], [0.10546875, 0.66015625], [0.0078125, 0.73046875], [0.24609375, 0.72265625], [0.1796875, 0.73828125], [0.77734375, 0.7578125], [0.34765625, 0.765625], [0.12890625, 0.78125], [0.25390625, 0.79296875], [0.17578125, 0.84765625], [0.4296875, 0.83203125], [0.35546875, 0.83984375], [0.45703125, 0.84765625], [0.5546875, 0.90234375], [0.76953125, 0.9140625], [0.39453125, 0.9296875], [0.359375, 0.93359375], [0.4453125, 0.95703125], [0.50390625, 0.96484375], [0.64453125, 0.96875], [0.9296875, 0.87890625], [0.74609375, 0.65234375], [0.70703125, 0.63671875], [0.70703125, 0.69140625], [0.65234375, 0.6875], [0.6171875, 0.70703125], [0.5625, 0.6953125], [0.4921875, 0.6953125], [0.44921875, 0.68359375], [0.43359375, 0.65625], [0.3984375, 0.63671875], [0.37109375, 0.62109375], [0.33203125, 0.609375], [0.328125, 0.56640625], [0.29296875, 0.52734375], [0.28125, 0.48828125], [0.2578125, 0.4140625], [0.24609375, 0.359375], [0.2578125, 0.3046875], [0.25390625, 0.25], [0.27734375, 0.1953125], [0.3203125, 0.13671875], [0.3984375, 0.08984375], [0.453125, 0.078125], [0.56640625, 0.08203125], [0.6171875, 0.09765625], [0.65625, 0.12890625], [0.65234375, 0.09375], [0.6953125, 0.12109375], [0.74609375, 0.15234375], [0.78515625, 0.1875], [0.81640625, 0.23046875], [0.82421875, 0.2734375], [0.8515625, 0.3828125], [0.83203125, 0.41796875], [0.84765625, 0.44921875], [0.82421875, 0.546875], [0.05078125, 0.11328125], [0.0625, 0.34375], [0.98046875, 0.64453125], [0.96875, 0.703125], [0.9140625, 0.78125], [0.90625, 0.83203125], [0.01953125, 0.08984375], [0.07421875, 0.15234375], [0.07421875, 0.375], [0.92578125, 0.9375], [0.96484375, 0.73828125], [0.93359375, 0.98046875], [0.98046875, 0.80859375], [0.3671875, 0.0], [0.83984375, 0.00390625], [0.7578125, 0.00390625], [0.1875, 0.02734375], [0.26953125, 0.02734375], [0.83984375, 0.03125], [0.33984375, 0.04296875], [0.0, 0.0546875], [0.875, 0.0625], [0.1796875, 0.0546875], [0.2578125, 0.10546875], [0.2109375, 0.2109375], [0.15625, 0.2109375], [0.1640625, 0.39453125], [0.15234375, 0.53125], [0.0078125, 0.57421875], [0.078125, 0.58203125], [0.046875, 0.609375], [0.0078125, 0.625], [0.16796875, 0.63671875], [0.1328125, 0.69140625], [0.8046875, 0.68359375], [0.19921875, 0.6875], [0.21875, 0.74609375], [0.8046875, 0.8046875], [0.6171875, 0.8515625], [0.39453125, 0.859375], [0.61328125, 0.88671875], [0.4140625, 0.8984375], [0.66015625, 0.8984375], [0.6640625, 0.93359375], [0.609375, 0.93359375], [0.78515625, 0.9609375], [0.296875, 0.97265625], [0.4140625, 0.98046875], [0.2734375, 0.9921875], [0.6015625, 0.9609375], [0.1953125, 0.91015625], [0.0703125, 0.52734375], [0.3125, 0.796875], [0.39453125, 0.80859375], [0.5625, 0.85546875], [0.24609375, 0.87890625]]
    var object_names_511 = ['lymphocyte', 'plasma', 'lymphocyte', 'lymphocyte', 'lymphocyte', 'epithelial', 'lymphocyte', 'plasma', 'epithelial', 'epithelial', 'lymphocyte', 'epithelial', 'epithelial', 'lymphocyte', 'lymphocyte', 'epithelial', 'epithelial', 'plasma', 'lymphocyte', 'plasma', 'lymphocyte', 'epithelial', 'lymphocyte', 'lymphocyte', 'plasma', 'lymphocyte', 'epithelial', 'plasma', 'epithelial', 'lymphocyte', 'lymphocyte', 'lymphocyte', 'lymphocyte', 'lymphocyte', 'plasma', 'plasma', 'plasma', 'lymphocyte', 'lymphocyte', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'lymphocyte', 'lymphocyte', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'epithelial', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue']

    if(display_4427)
    {
        feature_style = "4427" //malignant tissue
        points_511 = [[0.203125, 0.68359375], [0.58984375, 0.03125], [0.63671875, 0.0546875], [0.69921875, 0.12109375], [0.74609375, 0.13671875], [0.96484375, 0.13671875], [0.7265625, 0.20703125], [0.8125, 0.25], [0.01171875, 0.328125], [0.05859375, 0.3671875], [0.21484375, 0.39453125], [0.15625, 0.39453125], [0.33203125, 0.4921875], [0.8046875, 0.52734375], [0.625, 0.5234375], [0.98828125, 0.52734375], [0.8046875, 0.57421875], [0.609375, 0.578125], [0.58203125, 0.58984375], [0.2890625, 0.66015625], [0.88671875, 0.75390625], [0.3359375, 0.86328125], [0.1015625, 0.984375], [0.40234375, 0.5859375], [0.390625, 0.56640625], [0.3671875, 0.5390625], [0.328125, 0.4296875], [0.35546875, 0.41796875], [0.41796875, 0.4375], [0.45703125, 0.453125], [0.2265625, 0.56640625], [0.59765625, 0.74609375], [0.59375, 0.78125], [0.5859375, 0.8125], [0.625, 0.828125], [0.58984375, 0.88671875], [0.640625, 0.875], [0.69140625, 0.8984375], [0.703125, 0.93359375], [0.33984375, 0.9296875], [0.31640625, 0.89453125], [0.2890625, 0.90625], [0.265625, 0.9296875], [0.21484375, 0.9375], [0.16015625, 0.9375], [0.12890625, 0.94921875], [0.19140625, 0.07421875], [0.13671875, 0.06640625], [0.14453125, 0.015625], [0.1015625, 0.015625], [0.875, 0.18359375], [0.85546875, 0.21875], [0.90625, 0.203125], [0.90234375, 0.25390625], [0.53515625, 0.9140625], [0.23828125, 0.8046875], [0.25390625, 0.83984375], [0.1953125, 0.78125], [0.19921875, 0.74609375], [0.4921875, 0.421875], [0.53515625, 0.4140625], [0.29296875, 0.1953125], [0.484375, 0.09765625], [0.48828125, 0.0625], [0.5390625, 0.05859375], [0.8046875, 0.0], [0.84765625, 0.64453125], [0.79296875, 0.6484375], [0.73828125, 0.66015625], [0.73046875, 0.61328125], [0.69921875, 0.62890625], [0.69140625, 0.5625], [0.65234375, 0.55859375], [0.65625, 0.5859375], [0.5, 0.50390625], [0.515625, 0.53125], [0.55859375, 0.52734375], [0.58203125, 0.4765625], [0.578125, 0.44921875], [0.85546875, 0.41796875], [0.82421875, 0.421875], [0.7890625, 0.40625], [0.921875, 0.40625], [0.9296875, 0.44140625], [0.96875, 0.40234375], [0.77734375, 0.95703125], [0.8125, 0.984375], [0.84765625, 0.97265625], [0.89453125, 0.94140625], [0.92578125, 0.921875], [0.9921875, 0.81640625], [0.84375, 0.01953125], [0.9296875, 0.09765625], [0.40625, 0.15234375], [0.9375, 0.29296875], [0.9765625, 0.328125], [0.05859375, 0.4453125], [0.21875, 0.46875], [0.0546875, 0.5], [0.00390625, 0.51171875], [0.75, 0.5078125], [0.69140625, 0.51171875], [0.09375, 0.51953125], [0.0078125, 0.58203125], [0.4609375, 0.60546875], [0.48828125, 0.63671875], [0.1484375, 0.640625], [0.0703125, 0.6484375], [0.53125, 0.65625], [0.01171875, 0.69140625], [0.96484375, 0.71484375], [0.52734375, 0.703125], [0.05078125, 0.72265625], [0.390625, 0.75390625], [0.453125, 0.76953125], [0.0, 0.77734375], [0.40625, 0.80078125], [0.3359375, 0.80859375], [0.49609375, 0.8125], [0.453125, 0.83984375], [0.015625, 0.84765625], [0.96875, 0.8984375], [0.00390625, 0.98046875], [0.875, 0.046875], [0.95703125, 0.06640625], [0.40625, 0.12109375], [0.41015625, 0.3046875], [0.45703125, 0.34765625], [0.10546875, 0.41796875], [0.203125, 0.43359375], [0.03125, 0.46875], [0.0859375, 0.48828125], [0.20703125, 0.5078125], [0.08984375, 0.5625], [0.44140625, 0.58984375], [0.0390625, 0.5859375], [0.00390625, 0.62109375], [0.1875, 0.625], [0.51953125, 0.6328125], [0.03515625, 0.6484375], [0.54296875, 0.67578125], [0.9296875, 0.69140625], [0.05859375, 0.69140625], [0.01171875, 0.73828125], [0.5, 0.7734375], [0.01171875, 0.8046875], [0.98046875, 0.8671875], [0.015625, 0.88671875], [0.07421875, 0.90625], [0.046875, 0.98828125], [0.27734375, 0.05859375], [0.27734375, 0.2578125], [0.97265625, 0.296875], [0.3515625, 0.2890625], [0.4453125, 0.3203125], [0.953125, 0.34765625], [0.4765625, 0.3671875], [0.74609375, 0.48046875], [0.64453125, 0.484375], [0.70703125, 0.4921875], [0.421875, 0.75], [0.3125, 0.765625], [0.3828125, 0.8125], [0.53125, 0.81640625], [0.45703125, 0.8125], [0.875, 0.109375], [0.90234375, 0.12109375], [0.44140625, 0.9609375], [0.015625, 0.09375], [0.0703125, 0.1015625], [0.12890625, 0.1484375], [0.0, 0.15625], [0.08984375, 0.16796875], [0.14453125, 0.171875], [0.04296875, 0.2578125], [0.55078125, 0.9921875], [0.21875, 0.35546875], [0.2578125, 0.37109375], [0.24609375, 0.41796875], [0.8828125, 0.47265625], [0.16015625, 0.5], [0.16015625, 0.56640625], [0.921875, 0.58984375], [0.1328125, 0.69921875], [0.14453125, 0.7734375], [0.20703125, 0.8671875], [0.12109375, 0.87890625], [0.7578125, 0.0], [0.7734375, 0.01953125], [0.42578125, 0.07421875], [0.28515625, 0.57421875], [0.23828125, 0.62109375]]
        object_names_511 = ['epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'lymphocyte', 'epithelial', 'epithelial', 'lymphocyte', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'lymphocyte', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'lymphocyte', 'lymphocyte', 'lymphocyte', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'epithelial', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue', 'connectivetissue']
    }


    if(display_511 || display_4427){

        for (var i = 0; i < points_511.length; i++) {

        var left = points_511[i][0]
        var top = points_511[i][1]

//        console.log("Let us dive deeper")
//        console.log(left)
//        console.log(top)
//        console.log(containerOffsetLeft)
//        console.log(containerOffsetTop)
//        console.log(containerWidth)
//        console.log(containerHeight)

        var grid = 25 / 5;
        var text = object_names_511[i];
        var style = -1;
        var size = 0;

        allObjects.push({
            'height': 0,
            'width': 0,
            'left': left,
            'top': top,
            'text': text,
            'feature': feature_style,
            'size': size,
            'location': 0,
        });

        var newContent = document.createTextNode('*');
        var node = document.createElement("DIV");
        node.className = "resize-drag";
        node.id = guidGenerator();
        node.appendChild(newContent);
        node.style.color = nuclei_to_color_map[text]
        var init_size = 0;
        node.setAttribute('data-size', init_size);
        node.style.fontSize = sizeToFont(init_size);

        document.getElementById("resize-container").appendChild(node);

        // translate the element
        var rect = interact.getElementRect(node);
        var left_scaled = left*containerWidth;
        var top_scaled = top*containerHeight
        x_relative_shift = left_scaled - (rect.left - containerOffsetLeft)
        y_relative_shift = top_scaled - (rect.top - containerOffsetTop)
        node.style.transform = 'translate(' + x_relative_shift + 'px, ' + y_relative_shift + 'px)';

        addRow(text, size, location, style);
    }
    }

    var children = document.getElementsByClassName('resize-drag');

    if (children.length >= 1) {

        for (var i = 0; i < children.length; i++) {

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

        allObjects.push({
            'left': left,
            'top': top,
            'text': text,
            'feature': "hello",
        });
//        console.log(size, location, text);
        addRow(text, size, location, style);
    }
    }

    var image_id = document.getElementById('range-slider').getAttribute('image-id');
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
