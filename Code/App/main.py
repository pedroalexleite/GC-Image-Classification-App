#Imports
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import flask
import logging
import os
import tfmodel
from google.cloud import bigquery
from google.cloud import storage
import pandas as pd


#Set Up Logging
logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s',
                     datefmt='%Y-%m-%d %H:%M:%S')


PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT')
logging.info('Google Cloud project is {}'.format(PROJECT))


#Initialisation
logging.info('Initialising app')
app = flask.Flask(__name__)


logging.info('Initialising BigQuery client')
BQ_CLIENT = bigquery.Client()


BUCKET_NAME = PROJECT + '.appspot.com'
logging.info('Initialising access to storage bucket {}'.format(BUCKET_NAME))
APP_BUCKET = storage.Client().bucket(BUCKET_NAME)


logging.info('Initialising TensorFlow classifier')
TF_CLASSIFIER = tfmodel.Model(
    app.root_path + "/static/tflite/model.tflite",
    app.root_path + "/static/tflite/dict.txt"
)
logging.info('Initialisation complete')


#End Point Implementation
@app.route('/')
def index():
    return flask.render_template('index.html')


def generate_link(image_id):
    link = "https://storage.googleapis.com/bdcc_open_images_dataset/images/"+image_id+".jpg"
    return link


#FEITO PELO PROFESSOR
@app.route('/classes')
def classes():
    results = BQ_CLIENT.query(
    '''
        Select Description, COUNT(*) AS NumImages
        FROM `bdcc24project.openimages.image_labels`
        JOIN `bdcc24project.openimages.classes` USING(Label)
        GROUP BY Description
        ORDER BY Description
    ''').result()
    logging.info('classes: results={}'.format(results.total_rows))
    data = dict(results=results)
    return flask.render_template('classes.html', data=data)


#FEITO
@app.route('/relations')
def relations():
    results = BQ_CLIENT.query(
        '''
            SELECT Relation, COUNT(*) AS `Image count`
            FROM `bdcc24-project1.openimages.relations2`
            GROUP BY Relation
            ORDER BY Relation ASC
        '''
    ).result()
    relation_list = [{'Relation': row.Relation, 'Image count': row['Image count']} for row in results]

    return flask.render_template('relations.html', data={'relations': relation_list})


#FEITO
@app.route('/image_info')
def image_info():
    image_id = flask.request.args.get('image_id')

    sql_query = f'''
    SELECT DISTINCT j.Class AS Classes
    FROM `bdcc24-project1.openimages.joined3` j
    WHERE j.ImageId = "{image_id}"
    ORDER BY Classes ASC
    '''
    results1 = BQ_CLIENT.query(sql_query).result()
    results1_list = [list(row.values()) for row in results1]
    results1_df = pd.DataFrame(results1_list, columns=['Classes'])

    sql_query = f'''
    SELECT DISTINCT j.FinalRelation as Relations
    FROM `bdcc24-project1.openimages.joined3` j
    WHERE j.ImageId = "{image_id}"
    ORDER BY Relations ASC
    '''
    results2 = BQ_CLIENT.query(sql_query).result()
    results2_list = [list(row.values()) for row in results2]
    results2_df = pd.DataFrame(results2_list, columns=['Relations'])

    new_df = pd.DataFrame()
    new_df = pd.concat([new_df, results1_df], axis=1)
    new_df['Relations'] = results2_df

    link = "https://storage.googleapis.com/bdcc_open_images_dataset/images/"+image_id+".jpg"
    new_df.loc[0, 'Image'] = link

    return flask.render_template('image_info.html', image_id=image_id, data=new_df)


#FEITO
@app.route('/image_search')
def image_search():
    description = flask.request.args.get('description', default='')
    image_limit = flask.request.args.get('image_limit', default=10, type=int)

    sql_query = f'''
    Select ImageId from `bdcc24-project1.openimages.joined`
    WHERE Description = "{description}"
    ORDER BY ImageId
    LIMIT {image_limit}
    '''
    results = BQ_CLIENT.query(sql_query).result()

    results_list = [list(row.values()) for row in results]
    results_df = pd.DataFrame(results_list, columns=['ImageId'])
    results_df['Image'] = results_df['ImageId'].apply(lambda x: generate_link(x))
    results_count = len(results_df)

    return flask.render_template('image_search.html', description=description, image_limit=image_limit, results_count=results_count, data=results_df)


#FEITO
@app.route('/relation_search')
def relation_search():
    class1 = flask.request.args.get('class1', default='%')
    relation = flask.request.args.get('relation', default='%')
    class2 = flask.request.args.get('class2', default='%')
    image_limit = flask.request.args.get('image_limit', default=10, type=int)

    sql_query = f'''
    SELECT DISTINCT ImageId, Description1, Relation, Description2
    FROM `bdcc24-project1.openimages.joined3`
    WHERE Description1 LIKE "{class1}"
    AND Relation LIKE "{relation}"
    AND Description2 LIKE "{class2}"
    ORDER BY ImageId
    LIMIT {image_limit}
    '''
    results = BQ_CLIENT.query(sql_query).result()

    results_list = [list(row.values()) for row in results]
    results_df = pd.DataFrame(results_list, columns=['ImageId', 'Class 1', 'Relation', 'Class 2'])
    results_df['Image'] = results_df['ImageId'].apply(lambda x: generate_link(x))
    results_count = len(results_df)

    return flask.render_template('relation_search.html', class1=class1, relation=relation, class2=class2, image_limit=image_limit, results_count=results_count, data=results_df)


#FEITO PELO PROFESSOR
@app.route('/image_classify_classes')
def image_classify_classes():
    with open(app.root_path + "/static/tflite/dict.txt", 'r') as f:
        data = dict(results=sorted(list(f)))
        return flask.render_template('image_classify_classes.html', data=data)


#FEITO PELO PROFESSOR
@app.route('/image_classify', methods=['POST'])
def image_classify():
    files = flask.request.files.getlist('files')
    min_confidence = flask.request.form.get('min_confidence', default=0.25, type=float)
    results = []
    if len(files) > 1 or files[0].filename != '':
        for file in files:
            classifications = TF_CLASSIFIER.classify(file, min_confidence)
            blob = storage.Blob(file.filename, APP_BUCKET)
            blob.upload_from_file(file, blob, content_type=file.mimetype)
            blob.make_public()
            logging.info('image_classify: filename={} blob={} classifications={}'\
                .format(file.filename,blob.name,classifications))
            results.append(dict(bucket=APP_BUCKET,
                                filename=file.filename,
                                classifications=classifications))

    data = dict(bucket_name=APP_BUCKET.name,
                min_confidence=min_confidence,
                results=results)
    return flask.render_template('image_classify.html', data=data)


if __name__ == '__main__':
    # When invoked as a program.
    logging.info('Starting app')
    app.run(host='127.0.0.1', port=8080, debug=True)
