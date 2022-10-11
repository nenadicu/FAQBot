from HighPrecision import PrecisionService
from HighRecall import HighRecallService
from flask import Flask, request
import time
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
#cirs = CORS(app, resource={
#    r"/*":{
#        "origins":"*"
#    }
#})
#app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/question/<question>', methods=['GET'])
def start(question):
    top100 = hrs.get_best_n(question, 100)
    end = hprs.get_most_similar_question(top100, question)
    index = end[0][0]
    answer = hrs.get_answer(index)
    return answer

@app.route('/askQuestion', methods=['POST'])
def start1():
    question = request.form['question']
    top100 = hrs.get_best_n(question, 100)
    end = hprs.get_most_similar_question(top100, question)
    index = end[0][0]
    answer = hrs.get_answer(index)
    return answer

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    hrs = HighRecallService()
    hprs = PrecisionService()
    # start_time = time.time()
    #
    # top100 = hrs.get_best_n("How Much Is The Cost Of Home Cover? ", 100)
    # end = hprs.get_most_similar_question(top100, "How Much Is The Cost Of Home Cover? ")
    #
    # print("time elapsed: {:.2f}s".format(time.time() - start_time))
    # index = end[0][0]
    # answer = hrs.get_answer(index)
    # print(end)
    # print(answer)
    app.run()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
