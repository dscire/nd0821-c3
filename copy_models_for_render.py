import shutil

shutil.copyfile('./model/rfc_model.pkl', './model/rfc_model_deploy.pkl')
shutil.copyfile('./model/encoder.pkl', './model/encoder_deploy.pkl')
shutil.copyfile('./model/lb.pkl', './model/lb_deploy.pkl')
