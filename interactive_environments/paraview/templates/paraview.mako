<%namespace name="ie" file="ie.mako" />

<%
import os
import shutil
import hashlib

# Sets ID and sets up a lot of other variables
ie_request.load_deploy_config()
ie_request.attr.docker_port = 8777
ie_request.attr.import_volume = False

if ie_request.attr.PASSWORD_AUTH:
    m = hashlib.sha1()
    m.update( ie_request.notebook_pw + ie_request.notebook_pw_salt )
    PASSWORD = 'sha1:%s:%s' % (ie_request.notebook_pw_salt, m.hexdigest())
else:
    PASSWORD = "none"

DATASET_HID = hda.hid

input_tiff = ie_request.volume(hda.file_name, '/input/input.tiff', how='ro')

# Add all environment variables collected from Galaxy's IE infrastructure
ie_request.launch(volumes=[input_tiff],
    image=trans.request.params.get('image_tag', None),
    additional_ids=trans.request.params.get('additional_dataset_ids', None),
    env_override={
        'notebook_password': PASSWORD,
        'dataset_hid': DATASET_HID,
    }
)

## General IE specific
# Access URLs for the notebook from within galaxy.
notebook_server_url = ie_request.url_template('${PROXY_URL}/apps/')
notebook_access_url = ie_request.url_template('${PROXY_URL}/apps/Visualizer/')
notebook_login_url = ie_request.url_template('${PROXY_URL}/paraview')
notebook_ws_url = ie_request.url_template('${PROXY_URL}/ws')

%>
<html>
<head>
${ ie.load_default_js() }
</head>
<body>

<script type="text/javascript">
${ ie.default_javascript_variables() }
var notebook_access_url = '${ notebook_access_url }';
var notebook_server_url = '${ notebook_server_url }';
${ ie.plugin_require_config() }

// Load notebook
requirejs(['interactive_environments', 'plugin/paraview'], function(){
    load_notebook(ie_password, notebook_server_url, notebook_access_url);
});

</script>
<div id="main" width="100%" height="100%">
</div>
</body>
</html>
