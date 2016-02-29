// Javascript has backticks and multi-line strings? If you're using Chrome it does
// TODO: This could all be done using DOM manipulation or some template library
var user_template = `
    <div class="Area"> 
        <div class="R">
            <img src="/generic_person.png"/>
        </div>    
        <div class="text L textL">$TEXT</div>    
    </div>
`
var bernie_template = `
    <div class="Area"> 
        <div class="L">
            <a href="https://berniesanders.com">
                <img src="/bernie.jpg"/>
                <div class="tooltip">Senator Robot Bernie Sanders</div>
            </a>
        </div>    
        <div id='bernieResponse' class="text L textL">$TEXT</div>    
    </div>
`
$(function() {
    $("#userInput").focus();
});

// I trust Bernie not to output XSS, but safety is always the first priority
function escapeHtml(str) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
};

function askBernie() {
    // Display the user's message as a chat bubble
    var user_input = $('#userInput').val();
    $('#userInput').val('');
    $('.container').append(user_template.replace('$TEXT', escapeHtml(user_input)));
    deleteOverflow();

    // Kick off a repsonse from Bernie and display it when available
    // TODO: Handle timeouts from load
    $.ajax({
        url: "/ask_question",
        data: {
            'question': user_input
        },
        type: "GET",
        success: function(data) {
            $('.container').append(bernie_template.replace('$TEXT', escapeHtml(data)));
            deleteOverflow();
        },
        error: function(xhr) {
            $('.container').append('<div>Error, Robot Bernie Sanders is overloaded. Try again later.</div>');
        }
    });
    return false;
}

function deleteOverflow() {
    if ($('.Area').length > 5) {
        $('.Area')[0].remove()
    }
}
