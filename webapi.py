import flask
import bernie

app = flask.Flask(__name__, static_url_path='/static/')
model = bernie.load_model()

@app.route('/ask_question')
def ask_question():
    question = flask.request.args.get('question')
    answer = bernie.ask_bernie(model, question)
    return answer

@app.route('/')
def route_index():
    return app.send_static_file('bernie.html')

@app.route('/bernie.js')
def route_js():
    return app.send_static_file('bernie.js')

@app.route('/bernie.css')
def route_css():
    return app.send_static_file('bernie.css')

@app.route('/bernie.jpg')
def route_jpg():
    return app.send_static_file('bernie.jpg')

@app.route('/generic_person.png')
def route_person():
    return app.send_static_file('generic_person.png')

if __name__ == '__main__':
    app.run('0.0.0.0', port=8000, debug=True)
