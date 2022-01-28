from env import model_configs
from app import api, app

from flask_restful import Resource, reqparse
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

import uuid
import time
import json

from data_generator import vi_tokenize
from infer import infer_coliee_task3, list_split, init_state
from utils import Article

from utils_api.parser import get_config

cfg_service = get_config()
cfg_service.merge_from_file('cfgs/config.yaml')

SERVICE_IP = cfg_service.SERVICE.SERVICE_IP
SERVICE_PORT = cfg_service.SERVICE.SERVICE_PORT
MODEL_NAME = cfg_service.SERVICE.MODEL_NAME
LEGAL_CORPUS_FILE = cfg_service.SERVICE.LEGAL_CORPUS_FILE
with open('/workingspace/' + LEGAL_CORPUS_FILE, 'r', encoding='utf-8') as fo:
    LEGAL_CORPUS_DATA = json.load(fo)

# Get predictions
nlp = pipeline('question-answering', model=MODEL_NAME, tokenizer=MODEL_NAME)

model_init_states = {}
for m_name, model_info in tqdm(model_configs.items()):
    if 'tokenizer' in model_info and model_info['tokenizer'] == 'vi_tokenize':
        model_info['tokenizer'] = vi_tokenize

    print('Initialize the model')
    model_init_states[m_name] = init_state(**model_info)
    all_civil_code, data_args, tfidf_vectorizer, trainer, bert_tokenizer = model_init_states[
        m_name]
    print('Initialize the model successfully')

    tokenizer = model_info.get('tokenizer')
    topk = 150
print("Finish to load the model")


def find_corpus(corpus_data, law_id, article_id):
    for corpus in corpus_data:
        if corpus['law_id'] == law_id:
            return corpus['law_id'], corpus['articles'][int(article_id) - 1]

    return None, None
    
class serviceLegalTextRetrievalHandler(Resource):
    def get(self):
        return {'message': 'Service is alive.'}, 201
    
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('query', type=str) 

        args = parser.parse_args()
        
        if args['query'][-1] == '?':
            test_sents = [args['query']]
        else:
            test_sents = [args['query'] + " ?"]
        test_ids = [uuid.uuid4().hex]
        print(test_sents, test_ids)
        
        missing_ids_info = {}
        real_prediction = {}
        
        try:
            time_start = time.time()
            for m_name, _ in tqdm(model_configs.items()):
                predicted_labels, probs, c_code_pred_by_tfidf = infer_coliee_task3(sentence=test_sents, all_civil_code=all_civil_code,
                                                                                data_args=data_args,
                                                                                tfidf_vectorizer=tfidf_vectorizer,
                                                                                trainer=trainer, bert_tokenizer=bert_tokenizer,
                                                                                tokenizer=tokenizer, topk=topk)

                # np.array_split(predicted_labels, len(test_sents))
                predicted_labels = [x for x in list_split(predicted_labels, topk)]
                # np.array_split(probs, len(test_sents))
                probs = [x for x in list_split(probs, topk)]
                # np.array_split( c_code_pred_by_tfidf, len(test_sents))
                c_code_pred_by_tfidf = [
                    x for x in list_split(c_code_pred_by_tfidf, topk)]

                result = [[{"label": True if lb == 1 else False,
                            "scores": [float(probs[jj][i][j]) for j in range(probs[jj][i].shape[0])],
                        "id": test_ids[jj],
                            "sentence": s,
                            "civil_code_id": c_code_pred_by_tfidf[jj][i][1],
                            }
                        for i, lb in enumerate(predicted_labels[jj]) if lb == 1] for jj, s in enumerate(test_sents)]

                current_missing_ids = [[{"label":  False,
                                        "score": float(probs[jj][i][1]),
                                        "id": test_ids[jj],
                                        "civil_code_id": c_code_pred_by_tfidf[jj][i][1],
                                        }
                                        for i, lb in enumerate(predicted_labels[jj]) if lb == 0] for jj, s in enumerate(test_sents)]
                for negative_prediction in current_missing_ids:
                    negative_prediction.sort(
                        key=lambda info: info['score'], reverse=True)

                missing_ids_info[m_name] = current_missing_ids

                for jj, k in enumerate(test_ids):
                    if k not in real_prediction:
                        real_prediction[k] = set()
                    real_prediction[k] = real_prediction[k].union(
                        set([pred_infor['civil_code_id'] for pred_infor in result[jj]]))

                print("Finish inference on fine-tuned model {}, total time consuming: ".format(
                    m_name), time.time() - time_start)
                print(len(result))

            count_negative_add = 0
            for jj, k in enumerate(test_ids):
                if len(real_prediction[k]) == 0:
                    count_negative_add += 1
                    # pick 1 best score from negative prediction each model
                    for m_name, _ in model_configs.items():
                        real_prediction[k].add(
                            missing_ids_info[m_name][jj][0]['civil_code_id'])

            print("Total time consuming for {} samples: {} seconds => avg 1 sample in {} second".format(
                len(test_sents), time.time() - time_start,  (time.time() - time_start) / len(test_sents)))

            submit_result = []
            for k, v in real_prediction.items():
                relevant_a_s = []
                for relevant_a in v:
                    tmp_a = Article.from_string(relevant_a)
                    relevant_a_s.append(
                        {'law_id': tmp_a.l_id, 'article_id': tmp_a.a_id})
                submit_result.append({
                    'question_id': k,
                    'relevant_articles': relevant_a_s
                })
            print("Count negative addition = {}".format(count_negative_add))
            
            result = submit_result[0]["relevant_articles"]
            
            law_id, article_id = find_corpus(
                LEGAL_CORPUS_DATA, result[0]["law_id"], result[0]["article_id"])
            
            article_title = "Theo " + law_id + " tại " + \
                " về".join(article_id['title'].split(".")).lower()
            article_text = "<br/>".join(article_id["text"].split("\n"))
            
            qa_res = nlp({
                'question': args['query'], 
                'context': article_text
            })
            
            start_idx, end_idx = qa_res['start'], qa_res['end']
            
            return {'message': 'Successfully', 'result': json.dumps({"title": article_title, "text": article_text, "start_id": start_idx, "end_id": end_idx})}, 200
        except Exception as e:
            print(e)
            return {'message': 'No query selected'}, 404
        

api.add_resource(serviceLegalTextRetrievalHandler, '/api/legal_text_retrieval/')
if __name__ == "__main__":
    app.run(host=SERVICE_IP, threaded=True, port=SERVICE_PORT)
