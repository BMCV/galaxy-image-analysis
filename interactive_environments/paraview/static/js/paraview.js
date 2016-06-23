/**
 * Load an interactive environment (IE) from a remote URL
 * @param {String} password: password used to authenticate to the remote resource
 * @param {String} notebook_server_url: URL that should be POSTed to for login
 * @param {String} notebook_access_url: the URL embeded in the page and loaded
 *
 */
function load_notebook(password, notebook_server_url, notebook_access_url){
    $( document ).ready(function() {
        test_ie_availability(notebook_server_url, function(){
          append_notebook(notebook_access_url);
        });
    });
}
