#!/usr/bin/env python
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This tool creates an html visualization of a TensorFlow Lite graph.

Example usage:

python visualize.py foo.tflite foo.html
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
import shutil
import logging
import argparse
import webbrowser
import tempfile

from tflite_utils import DEFAULT_FLATC, DEFAULT_SCHEMA
from tflite_utils import check_schema_path, check_flatc_path

# A CSS description for making the visualizer
_CSS = """
<html>
<head>
<style>
body {font-family: sans-serif; background-color: #fa0;}
table {background-color: #eca;}
th {background-color: black; color: white;}
h1 {
  background-color: ffaa00;
  padding:5px;
  color: black;
}

svg {
  margin: 10px;
  border: 2px;
  border-style: solid;
  border-color: black;
  background: white;
}

div {
  border-radius: 5px;
  background-color: #fec;
  padding:5px;
  margin:5px;
}

.tooltip {color: blue;}
.tooltip .tooltipcontent  {
    visibility: hidden;
    color: black;
    background-color: yellow;
    padding: 5px;
    border-radius: 4px;
    position: absolute;
    z-index: 1;
}
.tooltip:hover .tooltipcontent {
    visibility: visible;
}

.edges line {
  stroke: #333;
}

text {
  font-weight: bold;
}

.nodes text {
  color: black;
  pointer-events: none;
  font-family: monospace;
  font-size: 12px;
}
</style>

<script src="https://d3js.org/d3.v4.min.js"></script>

</head>
<body>
"""

_D3_HTML_TEMPLATE = """
  <script>
    // Build graph data
    var graph = %s;

    var svg = d3.select("#subgraph%d")
    var width = svg.attr("width");
    var height = svg.attr("height");
    // Make the graph scrollable.
    svg = svg.call(d3.zoom().on("zoom", function() {
      svg.attr("transform", d3.event.transform);
    })).append("g");


    var color = d3.scaleOrdinal(d3.schemeDark2);

    var simulation = d3.forceSimulation()
        .force("link", d3.forceLink().id(function(d) {return d.id;}))
        .force("charge", d3.forceManyBody())
        .force("center", d3.forceCenter(0.5 * width, 0.5 * height));


    function buildGraph() {
      var edge = svg.append("g").attr("class", "edges").selectAll("line")
        .data(graph.edges).enter().append("path").attr("stroke","black").attr("fill","none")

      // Make the node group
      var node = svg.selectAll(".nodes")
        .data(graph.nodes)
        .enter().append("g")
        .attr("x", function(d){return d.x})
        .attr("y", function(d){return d.y})
        .attr("node_width", function(d){return d.node_width})
        .attr("transform", function(d) {
          return "translate( " + d.x + ", " + d.y + ")"
        })
        .attr("class", "nodes")
          .call(d3.drag()
              .on("start", function(d) {
                if(!d3.event.active) simulation.alphaTarget(1.0).restart();
                d.fx = d.x;d.fy = d.y;
              })
              .on("drag", function(d) {
                d.fx = d3.event.x; d.fy = d3.event.y;
              })
              .on("end", function(d) {
                if (!d3.event.active) simulation.alphaTarget(0);
                d.fx = d.fy = null;
              }));
      // Within the group, draw a box for the node position and text
      // on the side.

      var node_width = 150;
      var node_height = 30;

      node.append("rect")
          .attr("r", "5px")
          .attr("width", function(d) { return d.node_width; })
          .attr("height", node_height)
          .attr("rx", function(d) { return d.edge_radius; })
          .attr("stroke", "#000000")
          .attr("fill", function(d) { return d.fill_color; })
      node.append("text")
          .text(function(d) { return d.name; })
          .attr("x", 5)
          .attr("y", 20)
          .attr("fill", function(d) { return d.text_color; })
      // Setup force parameters and update position callback


      var node = svg.selectAll(".nodes")
        .data(graph.nodes);

      // Bind the links
      var name_to_g = {}
      node.each(function(data, index, nodes) {
        console.log(data.id)
        name_to_g[data.id] = this;
      });

      function proc(w, t) {
        return parseInt(w.getAttribute(t));
      }
      edge.attr("d", function(d) {
        function lerp(t, a, b) {
          return (1.0-t) * a + t * b;
        }
        var x1 = proc(name_to_g[d.source],"x") + proc(name_to_g[d.source],"node_width") / 2;
        var y1 = proc(name_to_g[d.source],"y") + node_height;
        var x2 = proc(name_to_g[d.target],"x") + proc(name_to_g[d.target],"node_width") / 2;
        var y2 = proc(name_to_g[d.target],"y");
        var s = "M " + x1 + " " + y1
            + " C " + x1 + " " + lerp(.5, y1, y2)
            + " " + x2 + " " + lerp(.5, y1, y2)
            + " " + x2  + " " + y2
      return s;
    });

  }
  buildGraph()
</script>
"""


class OpCodeMapper(object):
  """Maps an opcode index to an op name."""

  def __init__(self, data):
    self.code_to_name = {}
    for idx, d in enumerate(data["operator_codes"]):
      builtin_code = d["builtin_code"]
      if builtin_code == "CUSTOM":  # custom op
        self.code_to_name[idx] = d["custom_code"]
      else:  # proper builtin op
        self.code_to_name[idx] = d["builtin_code"]

  def __call__(self, x):
    if x not in self.code_to_name:
      s = "<UNKNOWN>"
    else:
      s = self.code_to_name[x]
    return "%s (%d)" % (s, x)


class DataSizeMapper(object):
  """For buffers, report the number of bytes."""

  def __call__(self, x):
    if x is not None:
      return "%d bytes" % len(x)
    else:
      return "--"


class TensorMapper(object):
  """Maps a list of tensor indices to a tooltip hoverable indicator of more."""

  def __init__(self, subgraph_data):
    self.data = subgraph_data

  def __call__(self, x):
    html = ""
    html += "<span class='tooltip'><span class='tooltipcontent'>"
    for i in x:
      tensor = self.data["tensors"][i]
      html += str(i) + " "
      html += tensor["name"] + " "
      html += str(tensor["type"]) + " "
      html += (repr(tensor["shape"]) if "shape" in tensor else "[]") + "<br>"
    html += "</span>"
    html += repr(x)
    html += "</span>"
    return html


def GenerateGraph(subgraph_idx, g, opcode_mapper):
  """Produces the HTML required to have a d3 visualization of the dag."""

  def TensorName(idx):
    return "t%d" % idx

  def OpName(idx):
    return "o%d" % idx

  edges, nodes = [], []
  tensor_names, op_names = [], []
  tensor_colors, op_colors = [], []
  first, second = {}, {}
  pixel_mult = 200  # TODO(aselle): multiplier for initial placement
  width_mult = 170  # TODO(aselle): multiplier for initial placement

  for tensor_index, tensor in enumerate(g["tensors"]):
    t_name = "(%d) %r %s" % (tensor_index, tensor["name"],
                        (repr(tensor["shape"]) if "shape" in tensor else "[]"))
    tensor_names.append((t_name, int(len(t_name) * 12/16*10 + 5)))
    tensor_colors.append("#fffacd")  # default tensor color, should be only parameters

  for op in g["operators"]:
    o_name = opcode_mapper(op["opcode_index"])
    op_names.append((o_name, int(len(o_name) * 12/16*10 + 5)))
    if o_name.startswith('XC_'):
      op_colors.append("#00a000")
    else:
      op_colors.append("#000000")

    # coloring intermediate tensors
    for tensor_index in op['outputs']:
      tensor_colors[tensor_index] = "#dddddd"

  #coloring input/output tensors
  for tensor_index in range(len(g['tensors'])):
    if tensor_index in g['inputs']:
      tensor_colors[tensor_index] = "#ccccff"
    elif tensor_index in g['outputs']:
      tensor_colors[tensor_index] = "#ffcccc"

  for op_index, op in enumerate(g["operators"]):

    x = width_mult
    for tensor_input_position, tensor_index in enumerate(op["inputs"]):
      if tensor_index not in first:
        first[tensor_index] = (
            (op_index - 0.5 + 1) * pixel_mult, x)
      x += tensor_names[tensor_index][1] + 10  # offset
      edges.append({
          "source": TensorName(tensor_index),
          "target": OpName(op_index), 
      })

    x = width_mult
    for tensor_output_position, tensor_index in enumerate(op["outputs"]):
      if tensor_index not in second:
        second[tensor_index] = (
            (op_index + 0.5 + 1) * pixel_mult, x)
      x += tensor_names[tensor_index][1] + 10  # offset
      edges.append({
          "target": TensorName(tensor_index),
          "source": OpName(op_index)
      })

    nodes.append({
        "id": OpName(op_index),
        "name": op_names[op_index][0],
        "text_color": "#eeeeee",
        "fill_color": op_colors[op_index],
        "edge_radius": 10,
        "x": pixel_mult,
        "y": (op_index + 1) * pixel_mult,
        "node_width": op_names[op_index][1]
    })

  for tensor_index, tensor in enumerate(g["tensors"]):
    initial_y = (
        first[tensor_index] if tensor_index in first
        else second[tensor_index] if tensor_index in second
        else (0, 0))

    nodes.append({
        "id": TensorName(tensor_index),
        "name": tensor_names[tensor_index][0],
        "text_color": "#000000",
        "fill_color": tensor_colors[tensor_index],
        "edge_radius": 1,
        "x": initial_y[1],
        "y": initial_y[0],
        "node_width": tensor_names[tensor_index][1]
    })

  graph_str = json.dumps({"nodes": nodes, "edges": edges})
  html = _D3_HTML_TEMPLATE % (graph_str, subgraph_idx)
  return html


def GenerateTableHtml(items, keys_to_print, display_index=True):
  """Given a list of object values and keys to print, make an HTML table.

  Args:
    items: Items to print an array of dicts.
    keys_to_print: (key, display_fn). `key` is a key in the object. i.e.
      items[0][key] should exist. display_fn is the mapping function on display.
      i.e. the displayed html cell will have the string returned by
      `mapping_fn(items[0][key])`.
    display_index: add a column which is the index of each row in `items`.
  Returns:
    An html table.
  """
  html = ""
  # Print the list of  items
  html += "<table><tr>\n"
  html += "<tr>\n"
  if display_index:
    html += "<th>index</th>"
  for h, mapper in keys_to_print:
    html += "<th>%s</th>" % h
  html += "</tr>\n"
  for idx, tensor in enumerate(items):
    html += "<tr>\n"
    if display_index:
      html += "<td>%d</td>" % idx
    # print tensor.keys()
    for h, mapper in keys_to_print:
      val = tensor[h] if h in tensor else None
      val = val if mapper is None else mapper(val)
      html += "<td>%s</td>\n" % val

    html += "</tr>\n"
  html += "</table>\n"
  return html


def CreateHtmlFile(tflite_input, html_output, *, schema, flatc_bin):
  """Given a tflite model in `tflite_input` file, produce html description."""

  # Convert the model into a JSON flatbuffer using flatc (build if doesn't
  # exist.
  if not os.path.exists(tflite_input):
    raise RuntimeError("Invalid filename %r" % tflite_input)
  if tflite_input.endswith(".tflite") or tflite_input.endswith(".bin"):

    # Run convert
    cmd = (
        flatc_bin + " -t "
        "--strict-json --defaults-json -o /tmp {schema} -- {input}".format(
            input=tflite_input, schema=schema))
    logging.info(f"Executing {cmd}")
    os.system(cmd)  # TODO: use subprocess.call
    real_output = ("/tmp/" + os.path.splitext(
        os.path.split(tflite_input)[-1])[0] + ".json")

    with open(real_output, 'r') as f:
      data = json.load(f)
  elif tflite_input.endswith(".json"):
    with open(tflite_input, 'r') as f:
      data = json.load(f)
  else:
    raise RuntimeError("Input file was not .tflite or .json")
  html = ""
  html += _CSS
  html += "<h1>TensorFlow Lite Model</h2>"

  data["filename"] = tflite_input  # Avoid special case
  toplevel_stuff = [("filename", None), ("version", None), ("description",
                                                            None)]

  html += "<table>\n"
  for key, mapping in toplevel_stuff:
    if not mapping:
      mapping = lambda x: x
    html += "<tr><th>%s</th><td>%s</td></tr>\n" % (key, mapping(data.get(key)))
  html += "</table>\n"

  # Spec on what keys to display
  buffer_keys_to_display = [("data", DataSizeMapper())]
  operator_keys_to_display = [("builtin_code", None), ("custom_code", None),
                              ("version", None)]

  for subgraph_idx, g in enumerate(data["subgraphs"]):
    # Subgraph local specs on what to display
    html += "<div class='subgraph'>"
    tensor_mapper = TensorMapper(g)
    opcode_mapper = OpCodeMapper(data)
    op_keys_to_display = [("inputs", tensor_mapper), ("outputs", tensor_mapper),
                          ("builtin_options", None), ("opcode_index",
                                                      opcode_mapper)]
    tensor_keys_to_display = [("name", None), ("type", None), ("shape", None),
                              ("buffer", None), ("quantization", None)]

    html += "<h2>Subgraph %d</h2>\n" % subgraph_idx

    # Inputs and outputs.
    html += "<h3>Inputs/Outputs</h3>\n"
    html += GenerateTableHtml(
        [{
            "inputs": g["inputs"],
            "outputs": g["outputs"]
        }], [("inputs", tensor_mapper), ("outputs", tensor_mapper)],
        display_index=False)

    # Print the tensors.
    html += "<h3>Tensors</h3>\n"
    html += GenerateTableHtml(g["tensors"], tensor_keys_to_display)

    # Print the ops.
    html += "<h3>Ops</h3>\n"
    html += GenerateTableHtml(g["operators"], op_keys_to_display)

    # Visual graph.
    html += "<svg id='subgraph%d' width='1600' height='900'></svg>\n" % (
        subgraph_idx,)
    html += GenerateGraph(subgraph_idx, g, opcode_mapper)
    html += "</div>"

  # Buffers have no data, but maybe in the future they will
  html += "<h2>Buffers</h2>\n"
  html += GenerateTableHtml(data["buffers"], buffer_keys_to_display)

  # Operator codes
  html += "<h2>Operator Codes</h2>\n"
  html += GenerateTableHtml(data["operator_codes"], operator_keys_to_display)

  html += "</body></html>\n"

  with open(html_output, "w") as f:
    f.write(html)


def main(tflite_input, html_output, *,
         no_browser=True, schema=DEFAULT_SCHEMA, flatc_bin=DEFAULT_FLATC):
  if html_output:
    html_path = html_output
  else:
    html_file = tempfile.NamedTemporaryFile(delete=False)
    html_path = html_file.name

  CreateHtmlFile(str(tflite_input), str(html_path),
                 schema=schema, flatc_bin=flatc_bin)

  if not no_browser:
    webbrowser.open_new_tab("file://" + os.path.realpath(html_path))

  if not html_output:
    html_file.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('tflite_input', help='Input .tflite file.')
  parser.add_argument('--html_output', required=False, default=None,
                      help='Output .html file. If not specified, a temporary file is created.')
  parser.add_argument('--no_browser',  action='store_true', default=False,
                      help='Do not open browser after the .html is created.')
  parser.add_argument('-v', '--verbose',  action='store_true', default=False,
                      help='Verbose mode.')
  args = parser.parse_args()
  tflite_input, html_output = args.tflite_input, args.html_output

  if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
  
  main(tflite_input, html_output, no_browser=args.no_browser)
