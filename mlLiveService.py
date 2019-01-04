from flask import Flask, jsonify, request
import Dataset as ds

app = Flask(__name__)
@app.route('/dataselect', methods = ['GET'])
def ingest():
    data = ds.Dataset()
    
    independentVar = request.args.get('independent')
    dataIndependent = data.dataSelection( selectionString=independentVar)

    dependentVar = request.args.get('dependent')
    print('ind___',independentVar)
    print('dep___', dependentVar)
    if not dependentVar :
        dependentVar = "Nothing"
    else:
        dataDependent = data.dataSelection( selectionString=dependentVar, extension="dependent")
    

    message = "I: " + independentVar + " D: " + dependentVar

    return message, 200
    
#Running the app

print('starting server...')
app.run(host = '0.0.0.0', port = 5000)