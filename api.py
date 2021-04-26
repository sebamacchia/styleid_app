import flask from Flask

app = Flask(__name__)


@app.route('/', methods=['GET'])
def api():
    return {
        'uderId': 1,
        'title': 'Flask React',
        'completed': False
    }
