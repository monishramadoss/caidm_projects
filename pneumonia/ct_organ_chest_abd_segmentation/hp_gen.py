import os
from pathlib  import Path
import importlib.util
from inspect import getmembers, isfunction, signature
import json

# print [o for o in getmembers(helloworld) if isfunction(o[1])]
model_str = {}
path = './models'
p = Path(path)
model_files = list(p.glob('**/model.py'))

f = open('train_temp.py', "rt")
train_template_str = f.read()


hp_selector = {
    'float' : 'hp.Float(name="{name}", min_value={min_value}, max_val={max_value}, step={step})',
    'int' : 'hp.Int(name="{name}", min_value={min_value}, max_value={max_value}, step={step})',
    'bool' : 'hp.Boolean(name="{name}", default={default})',
    'list' : 'hp.Choice(name="{name}", values={values})',
}

json_hp_schema = {
    'float' : {'type': 'float', 'min_value': None, 'max_value': None, 'step':0.1},
    'int' : {'type': 'int', 'min_value': None, 'max_value': None, 'step':1},
    'list' : {'type': 'list', 'values':None},
    'bool' : {'type' : 'bool', 'default': None}
}


for p in model_files:
    dir = p.parent
    export_dir = Path(os.path.join(dir, 'jmodels'))
    export_dir.mkdir(parents=True, exist_ok=True)

    name = p.parts[1]
    code = None
    hp_json = Path(os.path.join(dir, 'hp_json.json'))
    
    with p.open() as f:
        code = f.read()

    spec = importlib.util.spec_from_file_location(name, p)
    modulevar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulevar)
    params = {}
    hp_json_export = {}
    for o in getmembers(modulevar):
        if isfunction(o[1]) and o[0] == name:
            for k, p in signature(o[1]).parameters.items():
                if p.default != p.empty:
                    params[p.name] = p.default
                    
    
    define = str(name) + '(inputs, labels,\n\t\t'
    if hp_json.is_file():
        hp_json.touch()
        hp_json_export = {}
        for k, v in params.items():
            schema = json_hp_schema[str(type(v).__name__)].copy()
            for x, y in schema.items():
                if y is None:
                    schema[x] = v
            hp_json_export[k] = schema
        hp_json.open('wt').write(json.dumps(hp_json_export))
    
    
    hp_json_dict = json.loads(hp_json.open().read())
    for n, k in hp_json_dict.items():
        t = k['type']
        del k['type']
        k['name'] = n
        define += n + "=" + hp_selector[t].format(**k) + ',\n\t\t'
    define += ')'
    
    exp_dir = os.path.join(os.getcwd(), export_dir, 'exp')
    
    train_template_str = train_template_str % {'model':code, 'define': define, 'dis': str(exp_dir).replace('\\','\\\\'), 'project_name': name}
    
    file_path = export_dir / (name + '.py')
    with file_path.open('w', encoding='utf-8') as f:
        f.write(train_template_str)
    
    try:
        os.system('rm -rf "{0}/scripts"'.format(export_dir))
        os.system('jarvis script -jmodels ./models/{0}} -name ct_organ_seg_{0} -output_dir "{1}/scripts"'.format(name, export_dir))
    except:
        pass

