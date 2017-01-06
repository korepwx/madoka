var links = {
  'report': 'report/report.html',
  'log': 'logging.txt',
  'script': '_main.py',
  'browse': '.'
};

function activate_link(name) {
  $('nav li').removeClass('active');
  $('#' + name + '-nav').addClass('active');
  // Force the target to be fetched from server by adding a time stamp
  // to the URL.
  var timestamp = Date.now();
  var link = links[name] + '?t=' + timestamp;
  $('#content-page').attr('src', link);
}

function make_report() {
  bootbox.confirm({
    title: 'Generate Report',
    message: 'Do you want to re-generate the report?',
    buttons: {
      cancel: {
        label: 'Cancel',
        className: 'btn-default'
      },
      confirm: {
        label: 'Confirm',
        className: 'btn-primary'
      }
    },
    callback: function (confirmed) {
      if (confirmed) {
        // Keep a dialog open during the request
        wait_dialog = bootbox.dialog({
          message: '<div class="text-center">Generating Report...</div>',
          onEscape: false,
          closeButton: false
        });

        // Post the request to server
        $.ajax({
          url: '_api',
          method: 'POST',
          cache: false,
          data: '{"op": "make_report"}',
          dataType: 'json',
          contentType: 'application/json'
        }).done(function (result) {
          wait_dialog.modal('hide');
          if (result['error'] === 0) {
            activate_link('report');
          } else {
            bootbox.alert({
              title: 'Make Report Failure',
              message: 'Failed to make report: ' + result['message']
            });
          }
        });
      }
    }
  });
}

function delete_experiment() {
  bootbox.confirm({
    title: 'Delete Experiment',
    message: 'Do you want to delete the experiment?',
    buttons: {
      cancel: {
        label: 'Cancel',
        className: 'btn-default'
      },
      confirm: {
        label: 'Confirm',
        className: 'btn-danger'
      }
    },
    callback: function (confirmed) {
      if (confirmed) {
        // Keep a dialog open during the request
        wait_dialog = bootbox.dialog({
          message: '<div class="text-center">Deleting Experiment...</div>',
          onEscape: false,
          closeButton: false
        });

        // Post the request to server
        $.ajax({
          url: '_api',
          method: 'POST',
          cache: false,
          data: '{"op": "delete"}',
          dataType: 'json',
          contentType: 'application/json'
        }).done(function (result) {
          wait_dialog.modal('hide');
          if (result['error'] === 0) {
            window.location.href = '/';
          } else {
            bootbox.alert({
              title: 'Delete Experiment Failure',
              message: 'Failed to delete experiment: ' + result['message']
            });
          }
        });
      }
    }
  });
}

$(document).ready(function() {
  var url = window.location.href;
  var index = url.lastIndexOf('#');
  if (index >= 0) {
    var name = url.substr(index + 1);
    activate_link(name);
  } else {
    activate_link('report');
  }
});