import flask
import bernie

app = flask.Flask(__name__, static_url_path='/static/')
model = bernie.load_model()

@app.route('/ask_question')
def ask_question():
    question = flask.request.args.get('question')
    print 'Received question: {}'.format(question)
    answer = bernie.ask_bernie(model, question)
    print 'Returning answer: {}'.format(answer)
    return answer

@app.route('/')
def route_index():
    return app.send_static_file('bernie.html')

@app.route('/<path:path>')
def send_static_file(path):
    return flask.send_from_directory('static', path)

if __name__ == '__main__':
    app.run('0.0.0.0', port=8000)
