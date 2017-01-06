<template>
  <div id="app">
    <section class="todoapp">
      <header class="header">
        <h1>Experiments</h1>
        <input class="new-todo"
               autofocus autocomplete="off"
               placeholder="What experiments need to be found?"
               v-model="filterTitle">
      </header>
      <section class="main" v-show="todos.length" v-cloak>
        <input class="toggle-all" type="checkbox" v-model="allDone">

        <ul class="todo-list">
          <li v-for="todo in filteredTodos"
              class="todo"
              :key="todo.date">
            <div class="view">
              <input class="toggle" type="checkbox" v-model="todo.status" disabled="disabled">
              <label>
                <el-row>
                  <el-col :span="22">
                    <div style="margin: auto 0" class="caption">
                      <div style="font-size: 40px">{{todo.title}}</div>
                      <div style="font-size: 10px">{{todo.modifyTimeStr}}</div>
                    </div>
                  </el-col>
                  <el-col :span="2">
                    <div v-show="todo.active > 0" style="padding: 20px 0;" class="destroy">
                      <el-badge :value="todo.active" class="item">
                        <el-button @click="gotoDetail(todo)" type="text">
                          <i class="el-icon-arrow-right right-arrow"></i>
                        </el-button>
                      </el-badge>
                    </div>
                    <div v-show="todo.active == 0" style="padding: 20px 0;" class="destroy">
                      <el-button @click="gotoDetail(todo)" type="text">
                        <i class="el-icon-arrow-right right-arrow"></i>
                      </el-button>
                    </div>
                  </el-col>
                </el-row>
              </label>
            </div>
          </li>
        </ul>
      </section>
      <footer class="footer" v-show="todos.length" v-cloak>
        <span class="todo-count">
          <strong>{{ remaining }}</strong> {{ remaining | pluralize }}
        </span>
        <ul class="filters">
          <li><a href="#/all" :class="{ selected: visibility == 'all' }">All</a></li>
          <li><a href="#/active" :class="{ selected: visibility == 'active' }">Active</a></li>
          <li><a href="#/status" :class="{ selected: visibility == 'status' }">Completed</a></li>
        </ul>
        <button class="clear-status" @click="removeCompleted" v-show="todos.length > remaining">
          Clear status
        </button>
      </footer>
    </section>
    <el-dialog title="Historical Runs" v-model="detailVisible" size="large">
      <ol class="breadcrumb">
        <li v-for="title in detail.split('/')" class="breadcrumb-item active">{{ title }}</li>
      </ol>

      <el-row v-show="detailSelect != null" class="tool-row">
        <el-col :span="8" class="tagInput">
          <el-input placeholder="请输入内容" v-model="detailTag"></el-input>
        </el-col>
        <el-col :span="12">
          <el-button @click="addTag(detailSelect)" type="text">
            <i class="el-icon-edit tool-icon"></i>
          </el-button>
          <el-button @click="removeTodo(detailSelect)" type="text">
            <i class="el-icon-delete tool-icon"></i>
          </el-button>
          <el-button type="text">
            <a :href="detailSelect == null? '#' : (detailSelect.path + '/_navigation.html')" target="_blank">
              <i class="el-icon-share tool-icon"></i>
            </a>
          </el-button>
        </el-col>
      </el-row>

      <div class="row" v-for="(row, rindex) in detailTodo">
        <div class="col-lg-4 col-md-6 col-sm-12" v-for="(todo, cindex) in row">
          <div class="card card-block text-xs-center"
               :class="{'card-outline-primary' : todo.status == 0,
                        'card-outline-success' : todo.status == 1,
                        'card-outline-warning' : todo.status == 2}"
          >
            <blockquote class="card-blockquote">
              <h4 class="card-title">{{ todo.name }}</h4>
              <!--<h4 class="card-title"> Run {{ detailNum - (rindex * 3 + cindex) }}</h4>-->
              <div class="card-text">
                <div><small class="text-muted">Start Time: {{ todo.createTimeStr }}</small></div>
                <div><small class="text-muted">Last Update: {{ todo.modifyTimeStr }}</small></div>
              </div>
              <p class="card-tags">
                <el-tag v-for="(tag, tindex) in todo.tags"
                        :closable="false"
                        type="primary"
                        :close-transition="false"
                        @close="closeTag(todo, tindex)">
                  {{tag}}
                </el-tag>
              </p>
              <p v-if="todo.summary">{{ todo.summary }}</p>
              <a :href="todo.path + '/_navigation.html'" target="_blank" class="btn"
                 :class="{'btn-primary' : todo.status == 0,
                          'btn-success' : todo.status == 1,
                          'btn-warning' : todo.status == 2}">View Details</a>
            </blockquote>
          </div>
        </div>
      </div>

    </el-dialog>
    <footer class="info">
      <p>Written by <a href="http://evanyou.me">Wenxiao Chen</a></p>
      <p>Part of Madoka ML Toolkit</p>
    </footer>
  </div>
</template>


<script>
  function storageNameToTime(name) {
    var dateParts = name.match(/(\d{4})(\d{2})(\d{2}).(\d{2})(\d{2})(\d{2}).(\d{3})/);
    dateParts = dateParts.slice(1);
    console.log(dateParts);
    var dt = new Date(dateParts[0], dateParts[1]-1, dateParts[2], dateParts[3],
                      dateParts[4], dateParts[5], dateParts[6]);
    return dt;
  }
  function formatTime(t) {
    return t.toLocaleString();
  }
  function maxDate(a, b) {
    return a>b? a: b;
  }

  var STORAGE_KEY = 'todos-vuejs-2.0';
  var todoStorage = {
    fetch: function (app) {
      return app.$http.get('/_snapshot').then(function (res) {
        app.todos = [];
        app.collects = {};
        var title = null;
        for (var i in res.data) {
          var todo = res.data[i];
          todo.path = todo[0];
          todo.name = todo.path.substr(todo.path.lastIndexOf('/')+1);
          todo.tags = todo[1].filter(function (t) { return t != 'latest'; });
          todo.status = todo[2];
          todo.createTime = storageNameToTime(todo.name);
          todo.createTimeStr = formatTime(todo.createTime);
          todo.modifyTime = new Date(parseInt(todo[3]) * 1000);
          todo.modifyTimeStr = formatTime(todo.modifyTime);
          todo.title = todo.path.substr(0, todo.path.lastIndexOf('/'));
          todo.summary = '';
          if (app.collects[todo.title] == null)
            app.collects[todo.title] = [];
          app.collects[todo.title].push(todo);
        }
        for (var i in app.collects) {
          app.collects[i].sort(function(a, b) {
            return b.createTime - a.createTime;
          });
        }
        for (var i in app.collects) {
          var todo = {
            title: '',
            status: 0, //0 means running, 1 means no running
            active: 0,
            path: '',
            modifyTime: 0,
          };
          for (var j in app.collects[i]) {
            var item = app.collects[i][j];
            todo.title = item.title;
            if (item.status == 0)
              todo.active += 1;
            todo.path = item.path;
            todo.modifyTime = maxDate(todo.modifyTime, item.modifyTime);
          }
          todo.status = (todo.active == 0);
          todo.modifyTimeStr = formatTime(todo.modifyTime);
          app.todos.push(todo);
        }
        app.todos.sort(function(a, b) {
          return b.modifyTime - a.modifyTime;
        });
        todoStorage.uid = app.todos.length
      }, function (res) {
        console.log('Fetch Error');
      });
    },

    delete: function (app, todo) {
      return app.$http.post(todo.path + '/_api', {
        op: "delete"
      }).then(function (res) {
        console.log(res);
      }, function (res) {
        console.log('Delete Error');
      });
      //localStorage.setItem(STORAGE_KEY, JSON.stringify(todos))
    },

    addTags: function (app, todo, tags) {
      return app.$http.put(todo.path + '/_api', {
        op: "add_tags",
        tags: tags
      }).then(function (res) {
        console.log(res);
      }, function (res) {
        console.log('Error')
      })
    },

    removeTags: function (app, todo, tags) {
      return app.$http.put(todo.path + '/_api', {
        op: "remove_tags",
        tags: tags
      }).then(function (res) {
        console.log(res);
      }, function (res) {
        console.log('Error')
      })
    }


  };

  var filters = {
    all: function (todos) {
      return todos
    },

    active: function (todos) {
      return todos.filter(function (todo) {
        return !todo.status
      })
    },

    status: function (todos) {
      return todos.filter(function (todo) {
        return todo.status
      })
    },

    title: function (todos, search) {
      var title = search.split(':')[0];
      var tag = search.split(':')[1];
      return todos.filter(function (todo) {
        if (todo.path.indexOf(title) >= 0) {
          if (tag) {
            for (var t in todo.tags) {
              if (todo.tags[t].indexOf(tag) >= 0)
                return true
            }
          } else {
            return true
          }
        }
        return false
      })
    }
  }

  export default {
    data: function () {
      return {
        todos: [],
        collects: {},
        newTodo: '',
        visibility: 'all',
        filterTitle: '',
        detail: '',
        detailVisible: false,
        detailNum: 0,
        detailTag: '',
        detailSelect: null
      }
    },

    mounted: function () {
      todoStorage.fetch(this);
    },

    watch: {
      '$route': function (to, from) {
        var path = to.path.replace(/\//, '');
        if (filters[path]) {
          this.$set(this, 'visibility', path)
        } else {
          this.$set(this, 'visibility', 'all')
        }
      },

      todos: {
        handler: function (todos) {
        },
        deep: true
      }
    },

    computed: {
      filteredTodos: function () {
        var todos = filters[this.visibility](this.todos);
        return filters['title'](todos, this.filterTitle)
      },

      remaining: function () {
        return this.filteredTodos.length
      },

      allDone: {
        get: function () {
          return this.remaining === 0
        },
        set: function (value) {
          this.todos.forEach(function (todo) {
            todo.status = value
          })
        }
      },

      detailTodo: function () {
        var list = this.collects[this.detail];
        var res = [];
        var tmp = [];
        console.log(list);
        this.detailNum = 0;
        for (var todo in list) {
          /*if (tmp.length > 2) {
            res.push(tmp);
            tmp = [];
          }*/
          tmp.push(list[todo]);
          this.detailNum += 1;
        }
        if (tmp.length > 0) {
          res.push(tmp);
        }
        console.log(res);
        return res;
      },
    },

    filters: {
      pluralize: function (n) {
        return n === 1 ? 'experiment' : 'experiments'
      }
    },

    methods: {
      gotoDetail: function (todo) {
        this.detail = todo.title;
        this.detailVisible = true;
      },

      removeCompleted: function () {
        this.todos = filters.active(this.todos)
      },

      closeTag: function (todo, index) {
        if (index >= 0 && index < todo.tags.length) {
          todoStorage.removeTags(this, todo, [todo.tags[index]]);
          todo.tags.splice(index, 1);
        }
      },

      addTag: function (todo) {
        if (this.detailTag != '') {
          todoStorage.addTags(this, todo, [this.detailTag]);
          todo.tags.push(this.detailTag);
        }

      },

      removeTodo: function (todo) {
        console.log(this.collects[todo.title]);
        for (var i in this.collects[todo.title])
          if (this.collects[todo.title][i] == todo) {
            todoStorage.delete(this, todo);
            this.collects[todo.title].splice(i, 1);
            break;
          }
        var tmp = this.detail;
        this.detail = '';
        this.detail = tmp;
        this.detailSelect = null;
      },

      detailClass: function (row, index) {
        console.log(row, this.detailTodo[index].status)
        if (this.detailTodo[index].status == 0)
          return 'running-row';
        else if (this.detailTodo[index].status == 1)
          return 'success-row';
        else if (this.detailTodo[index].status == 2)
          return 'warning-row'
      },

      filterTag: function (value, row) {
        for (var i in row.tags)
          if (row.tags[i] == value)
            return true;
        return false;
      },

      row_click: function (row, e) {
        window.open(row.path + "/_navigation.html")
      },

      clickCard: function (todo) {
        this.detailSelect = todo == this.detailSelect ? null : todo;
        console.log('Click Card');
      }
    },

    directives: {
      'todo-focus': function (el, value) {
        if (value) {
          el.focus()
        }
      }
    }
  }
</script>

<style>
  @import 'css/todo.css';

  .el-table .running-row {
    background-color: rgba(32, 159, 255, .1);
  }

  .el-table .success-row {
    background-color: rgba(18, 206, 102, .2);
  }

  .el-table .warning-row {
    background-color: rgba(247, 186, 41, .1);
  }

  .right-arrow {
    font-size: 20px;
    color: #5dc2af
  }

  .tool-icon {
    font-size: 20px;
    color: #8492a6;
    margin: 0 0;
    vertical-align: middle
  }

  .tool-row {
    margin: 20px 0 20px 0;
  }

  .tagInput {
    margin: 0 20px 0 0;
  }
</style>
