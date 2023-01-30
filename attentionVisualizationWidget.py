import baukit
from baukit import Widget, Property, Trigger, Textbox, show

import torch, numpy as np
import os, re, json
# from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm


class TokenVizWidget(Widget):
    """
    """
    def __init__(self, text, token_attn, default_display=None, 
                 current_layer=None, current_head=None, def_range=[-1,1], **kwargs):
        super().__init__(**kwargs)
        
        self.text = Property(text)
        self.last_clicked = Property(None)       
        self.token_attn = Property(token_attn)
        self.default_display = Property(default_display)
        self.current_layer = Property(current_layer)
        self.current_head = Property(current_head)
        
        if self.default_display is not None:
            self.dep_min, self.dep_max  = def_range[0] - 0.01, def_range[1] + 0.01
            self.colors_matrix = Property(self.color_sample(self.default_display, cm.bwr))
            
    def normalize(self,x):
        return TwoSlopeNorm(0,self.dep_min, self.dep_max)(x).data
    
    def color_sample(self,x,cm):
        return (cm(self.normalize(np.array(x)))[:,:3] * 255).tolist()
                
    def widget_js(self):
        return '''
        class MyWidget {
          constructor(element, model) {
            this.element = element;
            this.model = model;
            this.tokens = element.querySelectorAll('span');
            
            // Set event listeners for each token span
            this.tokens.forEach((token, i) => {
              token.addEventListener('mouseout', () => {
                var last_clicked = model.get('last_clicked');
                if (last_clicked == null){
                  this.hover_set_background_color(-1);
                  //this.hover_set_border(i,false);
                  this.static_set_background_color(this.tokens.length, true);
                }
              });
              token.addEventListener('mouseover', () => {
                var last_clicked = model.get('last_clicked');
                if (last_clicked == null){
                  this.hover_set_background_color(i, true);
                  this.hover_set_border(i,true);
                }
              });
              token.addEventListener('click', () => {
                var last_clicked = model.get('last_clicked');
                if (i == last_clicked){
                  this.set_last_clicked(null);
                  this.hover_set_background_color(-1);
                  this.static_set_background_color(this.tokens.length, true);
                }
                else{
                  if (last_clicked != null){
                    this.static_set_background_color(i,false);
                    this.hover_set_background_color(i, true);
                  }
                  this.set_last_clicked(i);
                  this.hover_set_background_color(i, false);
                  this.hover_set_border(i,true);
                }
              });
            });
            
            // Initialize token color to token defaults
            this.static_set_background_color(this.tokens.length, true)
            
            //updates variables accessible through interaction/python
            this.model.on('last_clicked', () => {
              this.set_last_clicked(model.get('last_clicked'), true);
            });          
          };
          
          // Define functions for event triggers
          set_last_clicked(token_i, python_notified) {
            if (!python_notified) {
              this.model.set('last_clicked', token_i);
            }
          };
          
          //Define rules for coloring background of token spans during hover
          hover_set_background_color(i, add_title){
            for (var j = 0; j <= i; j++) {
              var s = this.tokens[j];
              var h = this.model.get('current_head');
              var m = 2;
              var opacity = this.model.get('token_attn')[i][j] * m;
              if (add_title){
                s.title += `\nAttn: ${(opacity/m).toFixed(4)}\nToken Pos: ${j}`;
              }
              if (opacity/m > 0.02){
                s.style.border = "1px solid rgba(128,128,128,0.5)"; //gray border
              }
              else{
                s.style.border = "1px solid rgba(128,128,128,0)"; //no border
              }
              s.style.backgroundColor = `hsla(${15 * h}, 100%, 50%,${opacity})`
              //s.style.backgroundColor = `rgba(255, 0, 0, ${opacity})`;
            }
            for (var j = i + 1; j < this.tokens.length; j++) {
              var s = this.tokens[j];
              s.style.backgroundColor = null;
              s.title = '';
              s.style.border = "1px solid rgba(128,128,128,0)";
            }
          };
          
          hover_set_border(i,turn_on){
            // Set border of current hover token
            var s = this.tokens[i];
            // Tick marks are like python's f-string, or .format
            s.style.border = `${turn_on ? "1px solid rgba(0,0,0,1)" : "1px solid rgba(128,128,128,0)"}`;
          };
          
          // Define rules for coloring background with static values
          static_set_background_color(i, turn_on, python_notified){
              if (!python_notified){
                  for (var j = 0; j < i; j++){
                      var s = this.tokens[j];
                      var c_val = model.get('colors_matrix')[j];
                      var mag = model.get('default_display')[j];
                      if (turn_on && Math.abs(mag) > 0.01){
                          s.style.backgroundColor = `rgba(${c_val[0]}, ${c_val[1]}, ${c_val[2]},1)`;
                          s.style.border = "1px solid rgba(128,128,128,0.5)";
                          s.title += `Token Default Value: ${mag.toFixed(5)}`;
                      }
                      else{
                          s.style.backgroundColor = null;
                          s.style.border = "1px solid rgba(128,128,128,0)";
                          s.title = '';
                      }
                  }
              }
          };         
        };
        token_widget = new MyWidget(element, model);
        '''
    def widget_html(self):
        return (
            f'<div id="{self.view_id()}">' +
            ''.join([f'<span style="border:1px solid rgba(128,128,128,0)">{w}</span>' if w != '\n' else f'<span style="border:1px solid rgba(128,128,128,0)">{w}</span><br>' for w in self.text]) +
            '</div>')

class AttnHeadSelectorWidget(Widget):
    def __init__(self, n_layers=28,n_heads=16,head_groupings={},**kwargs):
        super().__init__(**kwargs)
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.current_layer = Property(0)
        self.current_head = Property(0)
        self.head_groupings = Property(list(head_groupings.values()))
        self.n_head_groups = Property(len(head_groupings.keys()))
        self.head_group_names = list(head_groupings.keys())
        
    def widget_js(self):
        return '''
        class MyWidget {
          constructor(element, model) {
            this.element = element;
            this.model = model;
            this.selects = element.querySelectorAll('select');
            
            this.buttons = element.querySelectorAll('button');
            this.groups = element.querySelectorAll('div');
            this.n_groups = model.get('n_head_groups');
            
            // Set event listeners for layer & head select & display type 
            this.selects[0].addEventListener('change', () => {
                var v = parseInt(this.selects[0].value);
                this.set_select_value(0, v);
            });
            this.selects[1].addEventListener('change', () => {
                var v = parseInt(this.selects[1].value);
                this.set_select_value(1, v);
            });

            this.model.on("current_layer", ()=>{
                this.set_select_value(0, model.get('current_layer'),true);
            });
            this.model.on("current_head", ()=>{
                this.set_select_value(1, model.get('current_head'),true);
            });

            this.buttons.forEach((button,i) => {
              button.addEventListener('click', () => {
                
                if (button.style.backgroundColor == ''){
                  button.style.backgroundColor = `rgba(${173},${216},${230},0.7)`;
                }
                else{
                  button.style.backgroundColor = '';
                }
                
                if (i < this.n_groups){
                  var spans = this.groups[i].querySelectorAll('button')
                  spans.forEach((span,i)=>{
                    if (span.style.display == 'none'){
                      span.style.display = 'inline';
                    }
                    else if (span.style.display != 'none'){
                      span.style.display = 'none';                  
                    }
                  })
                }
                else{
                  var lh_array = button.value.replace('(', '').replace(')','').split(',').map((x) => parseInt(x));
                  //console.log('button clicked:',lh_array);
                  this.set_select_value(0, lh_array[0]);
                  this.selects[0].value = `${lh_array[0]}`;
                  this.set_select_value(1, lh_array[1]);
                  this.selects[1].value = `${lh_array[1]}`;
                }
              });
            });
          };
                    
          set_select_value(i,v, python_notified){
            if (!python_notified){
                if (i==0){
                    this.model.set("current_layer",v);
                }
                else if (i==1){
                    this.model.set("current_head",v);
                }
            }
          };
        };
    
        attn_head_selector_widget = new MyWidget(element, model)
        '''
    
    def widget_html(self):
        return (
        f'<div id="{self.view_id()}">' + 
            '<span>' + 
                '<label for="layer_selection">Choose a Layer:</label>' + 
                '<select name="layer_selection" id="layer_selection">' + 
                ''.join([f'<option value={i}>Layer {i}</option>' for i in range(self.n_layers)]) + 
                '</select>'+

                '<label for="head_selection">&nbsp&nbsp Choose a Head:</label>' + 
                '<select name="head_selection" id="head_selection">' + 
                ''.join([f'<option value={i}>Head {i}</option>' for i in range(self.n_heads)]) + 
                '</select>'+
            '</span>'+
            '<br>'+
            ''.join([f'<button>Head Group {i}: {self.head_group_names[i]}</button>' for i in range(self.n_head_groups)])+
            ''.join([f'<div id="group_{j}">' + 
            ''.join([f'<button style="display:none" value="{x}">{x}</button>' for i,x in enumerate(self.head_groupings[j])])+
            '</div>' for j in range(self.n_head_groups)])+            
        '</div>'
        )
