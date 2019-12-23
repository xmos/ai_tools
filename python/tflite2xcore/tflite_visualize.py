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

import json
import os
import sys
import shutil
import logging
import argparse
import webbrowser
import tempfile

from tflite2xcore.serialization.flatbuffers_io import create_dict_from_model
from tflite2xcore import read_flatbuffer

# A CSS description for making the visualizer
_CSS = """<head>
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
<script src="https://d3js.org/d3.v5.min.js"></script>
</head>
"""
# TODO: reference d3.js script locally

_D3_HTML_TEMPLATE = """<script>
function buildGraph() {
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
      .attr("height", function(d) { return d.node_height; })
      .attr("rx", function(d) { return d.edge_radius; })
      .attr("stroke", "#000000")
      .attr("fill", function(d) { return d.fill_color; })
  let text = node.append("text")
      .text("")
      .attr("x", 5)
      .attr("y", 1)
      .attr("fill", function(d) { return d.text_color; });
  text.selectAll("text")
      .data(d => d.name.split("\\n"))
      .enter()
      .append("tspan")
      .attr("class", "text")
      .text(d => d)
      .attr("x", 5)
      .attr("dy", 15);

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


class OpCodeMapper():
    """Maps an opcode index to a text representation."""

    def __init__(self, data):
        self.idx_to_name = [
            d["custom_code"] if d["builtin_code"] == "CUSTOM" else d["builtin_code"]
            for d in data["operator_codes"]
        ]

    def __call__(self, x):
        s = self.idx_to_name[x] if x < len(self.idx_to_name) else "UNKNOWN"
        return f"({x}) {s}"


class DataSizeMapper():
    """For buffers, report the number of bytes."""

    def __call__(self, x):
        return "--" if x is None else f"{len(x):d} bytes"


class TensorMapper():
    """Maps a tensor index to a text representation."""

    def __init__(self, subgraph, node_text=False):
        self.tensors = subgraph["tensors"]
        self.node_text = node_text

    def __call__(self, idx):
        tensor = self.tensors[idx]
        if self.node_text:
            return (f"{tensor['name']}"
                    + "\n"
                    + f"({idx:d}) <{tensor['type']}> {tensor['shape']}")
        else:
            return f"({idx:d}) {tensor['name']} &lt{tensor['type']}&gt {tensor['shape']} <br>"


class TensorTooltipMapper():
    """Maps a list of tensor indices to a tooltip hoverable indicator of more."""

    def __init__(self, subgraph):
        self.tensor_mapper = TensorMapper(subgraph)

    def __call__(self, idx_list):
        html = "<span class='tooltip'><span class='tooltipcontent'>"
        for idx in idx_list:
            html += self.tensor_mapper(idx)
        html += f"</span>{idx_list}</span>"
        return html


def GenerateGraph(subgraph_idx, g, opcode_mapper):
    """Produces the HTML required to have a d3 visualization of the dag."""

    def TensorID(idx):
        return f"t{idx:d}"

    def OperatorID(idx):
        return f"o{idx:d}"

    def NodeWidth(node_text):
        return int(max(len(line) * 12/16*10 + 5 for line in node_text.split("\n")))

    def NodeHeight(node_text):
        return node_text.count("\n") * 15 + 25

    edges, nodes = [], []
    tensor_nodes_info, op_nodes_info = [], []
    first, second = {}, {}
    pixel_mult = 200  # TODO(aselle): multiplier for initial placement
    width_mult = 170  # TODO(aselle): multiplier for initial placement

    tensor_mapper = TensorMapper(g, node_text=True)

    for tensor_index, tensor in enumerate(g["tensors"]):
        t_node_text = f"({tensor_index:d}) {tensor['name']} {tensor['shape']}"
        t_node_text = tensor_mapper(tensor_index)
        tensor_nodes_info.append({
            'text': t_node_text,
            'width': NodeWidth(t_node_text),
            'height': NodeHeight(t_node_text),
            'color': "#fffacd"  # default tensor color, should be only for parameters
        })

    for op in g["operators"]:
        o_node_text = opcode_mapper(op["opcode_index"])
        op_nodes_info.append({
            'text': o_node_text,
            'width': NodeWidth(o_node_text),
            'height': NodeHeight(o_node_text),
            'color': "#00a000" if ' XC_' in o_node_text else "#000000"
        })

        # coloring intermediate tensors
        for tensor_index in op['outputs']:
            tensor_nodes_info[tensor_index]['color'] = "#dddddd"

    # coloring input/output tensors
    for tensor_index in range(len(g['tensors'])):
        if tensor_index in g['inputs']:
            tensor_nodes_info[tensor_index]['color'] = "#ccccff"
        elif tensor_index in g['outputs']:
            tensor_nodes_info[tensor_index]['color'] = "#ffcccc"

    for op_index, op in enumerate(g["operators"]):
        x = width_mult
        for tensor_index in op["inputs"]:
            if tensor_index not in first:
                first[tensor_index] = ((op_index - 0.5 + 1) * pixel_mult, x)
            x += tensor_nodes_info[tensor_index]['width'] + 10  # offset
            edges.append({
                "source": TensorID(tensor_index),
                "target": OperatorID(op_index),
            })

        x = width_mult
        for tensor_index in op["outputs"]:
            if tensor_index not in second:
                second[tensor_index] = ((op_index + 0.5 + 1) * pixel_mult, x)
            x += tensor_nodes_info[tensor_index]['width'] + 10  # offset
            edges.append({
                "target": TensorID(tensor_index),
                "source": OperatorID(op_index)
            })

        nodes.append({
            "id": OperatorID(op_index),
            "name": op_nodes_info[op_index]['text'],
            "text_color": "#eeeeee",
            "fill_color": op_nodes_info[op_index]['color'],
            "edge_radius": 10,
            "x": pixel_mult,
            "y": (op_index + 1) * pixel_mult,
            "node_width": op_nodes_info[op_index]['width'],
            "node_height": op_nodes_info[op_index]['height']
        })

    for tensor_index, tensor in enumerate(g["tensors"]):
        initial_y = (
            first[tensor_index] if tensor_index in first
            else second[tensor_index] if tensor_index in second
            else (0, 0))

        nodes.append({
            "id": TensorID(tensor_index),
            "name": tensor_nodes_info[tensor_index]['text'],
            "text_color": "#000000",
            "fill_color": tensor_nodes_info[tensor_index]['color'],
            "edge_radius": 1,
            "x": initial_y[1],
            "y": initial_y[0],
            "node_width": tensor_nodes_info[tensor_index]['width'],
            "node_height": tensor_nodes_info[tensor_index]['height']
        })

    graph_str = json.dumps({"nodes": nodes, "edges": edges}, indent=2)
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
    indent = " " * 2

    # Print the list of  items
    html = "<table>"
    html += "<tr>\n"
    if display_index:
        html += f"{indent}<th>index</th>\n"
    for h, mapper in keys_to_print:
        html += f"{indent}<th>{h}</th>\n"
    html += "</tr>"

    # print rows
    for idx, tensor in enumerate(items):
        html += "<tr>\n"
        if display_index:
            html += f"{indent}<td>{idx}</td>\n"
        # print tensor.keys()
        for h, mapper in keys_to_print:
            val = tensor[h] if h in tensor else None
            val = val if mapper is None else mapper(val)
            html += f"{indent}<td>{val}</td>\n"

        html += "</tr>"
    html += "</table>\n\n"
    return html


def CreateHtmlFile(tflite_input, html_output):
    """Given a tflite model in `tflite_input` file, produce html description."""

    if not os.path.exists(tflite_input):
        raise RuntimeError(f"Invalid filename {tflite_input}")

    model = read_flatbuffer(tflite_input)
    data = create_dict_from_model(model)

    indent = " " * 2

    html = "<html>\n"
    html += _CSS
    html += "\n<body>\n"
    html += "<h1>TensorFlow Lite Model</h1>\n"

    data["filename"] = tflite_input  # Avoid special case
    toplevel_stuff = [("filename", None),
                      ("version", None),
                      ("description", None)]

    html += "<table>"
    for key, mapping in toplevel_stuff:
        html += "<tr>\n"
        html += f"{indent}<th>{key}</th>\n"
        key = key if mapping is None else mapping(data.get(key))
        html += f"{indent}<td>{key}</td>\n"
        html += "</tr>"
    html += "</table>\n"

    # Spec on what keys to display
    buffer_keys_to_display = [("data", DataSizeMapper())]
    operator_keys_to_display = [("builtin_code", None),
                                ("custom_code", None),
                                ("version", None)]

    for subgraph_idx, g in enumerate(data["subgraphs"]):
        # Subgraph local specs on what to display
        html += "\n<div class='subgraph'>"
        tensor_mapper = TensorTooltipMapper(g)
        opcode_mapper = OpCodeMapper(data)
        op_keys_to_display = [("inputs", tensor_mapper),
                              ("outputs", tensor_mapper),
                              ("opcode_index", opcode_mapper),
                              ("builtin_options", None),
                              ("custom_options", None)]
        tensor_keys_to_display = [("name", None),
                                  ("type", None),
                                  ("shape", None),
                                  ("buffer", None),
                                  ("quantization", None)]

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
        html += "</div>\n\n"

    # Buffers have no data, but maybe in the future they will
    html += "<h2>Buffers</h2>\n"
    html += GenerateTableHtml(data["buffers"], buffer_keys_to_display)

    # Operator codes
    html += "<h2>Operator Codes</h2>\n"
    html += GenerateTableHtml(data["operator_codes"], operator_keys_to_display)

    html += "</body>\n</html>\n"

    with open(html_output, "w") as f:
        f.write(html)


def main(tflite_input, html_output, *, no_browser=True):
    if html_output:  # TODO: do this with a context manager
        html_path = html_output
    else:
        html_file = tempfile.NamedTemporaryFile(delete=False)
        html_path = html_file.name

    CreateHtmlFile(str(tflite_input), str(html_path))

    if not no_browser:
        webbrowser.open_new_tab("file://" + os.path.realpath(html_path))

    if not html_output:
        html_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tflite_input', help='Input .tflite file.')
    parser.add_argument('-o', '--html_output', required=False, default=None,
                        help='Output .html file. If not specified, a temporary file is created.')
    parser.add_argument('--no_browser', action='store_true', default=False,
                        help='Do not open browser after the .html is created.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Verbose mode.')
    args = parser.parse_args()
    tflite_input, html_output = args.tflite_input, args.html_output

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    main(tflite_input, html_output, no_browser=args.no_browser)
