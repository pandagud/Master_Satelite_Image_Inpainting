from src.config_default import TrainingConfig
def update_config(args,config):
    # Instantiate the parser
    d = dict(arg.split(':') for arg in args)
    c = config.__dict__
    for key in d.keys():
        if key in c.keys():
            newValue = d[key]
            localType = c[key]
            localType = type(localType)
            if localType==int:
                c[key] = int(newValue)
            elif localType==float:
                c[key]=float(newValue)
            elif localType==bool:
                if newValue=='False':
                    c[key] = False
                else :
                    c[key] =True
            else:
                c[key]=newValue
    new_config = TrainingConfig(**c)
    return new_config