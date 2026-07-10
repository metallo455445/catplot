#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import fletcher.shapes: house, hexagon
#import fletcher.shapes
#set page(width: auto, height: auto, margin: 5mm, fill: white)
#set text(font: "New Computer Modern")

#let blob(pos, label, tint: white, ..args) = node(
	pos, align(center, label),
	width: 28mm,
	fill: tint.lighten(60%),
	stroke: 1pt + tint.darken(20%),
	corner-radius: 5pt,
	..args,
)

#diagram(
  spacing: 8pt,
  cell-size: (8mm, 10mm),
  edge-stroke: 1pt,
  edge-corner-radius: 5pt,
  mark-scale: 70%,

  blob((.25,1), [Original Image], tint: red),

  edge((0.23, 1), (0.23, 2.1), (1, 2.1), (1, 3), "-|>"),
  edge((0.27, 1), (0.27, 2), (2, 2), (2, 3), "-|>"),
  edge((0.31, 1), (0.31, 1.9), (3, 1.9), (3, 3), "-|>"),

  //retangle around the three methods
  node(enclose: (<A>, <B>, <C>), shape: shapes.rect, fill: red.lighten(90%), stroke: 1pt + red),

  blob((1,3), [Binary], name:<A>,tint: yellow),
  blob((2,3), [Adaptive], name:<B>,tint: yellow),
  blob((3,3), [Gaussian], name:<C>, tint: yellow),

  edge((0.19, 1), (0.19, 4), "-|>"),

  blob((.19,4), [Edges], tint: blue),

  edge((1,3), (1,3.9), (.2,3.9),"-|>"),
  edge((2,3), (2,4), (.4,4),"-|>"),
  edge((3,3), (3,4.1), (.6,4.1),"-|>"),

  edge("-|>"),

  blob((.19,5), [Contours], tint: blue),

  edge("-|>"),

  blob((.19,6), [Contours sorted], tint: blue),

  edge([first n],"-|>"),

  blob((1.75,6), [Top contours], tint: green),

  edge("--|>"),

  blob((3.25,6), [Noise], tint: red),

  edge((3.25,6), (3.25,7), (2,7), (2,6), [user selection], "--|>"),

  edge((1.75,6), (1.75,7.5), "-|>"),

  blob((1.75, 7.75), [For each contour], tint: purple, shape: hexagon),

  edge((1.75, 7.75), (3.25, 7.75), (3.25, 8.75), "-|>"),

  edge((3.25, 8.75), (1.75, 8.75), (1.75, 7.75), "-|>"),

  blob((3.25, 8.75), [Get coordinates], tint: blue),

  edge("-|>"),

  blob((3.25, 10.25), [write in 'coord.txt'], tint: yellow),

  edge((3.25, 10.25), (1.75, 10.25), (1.75, 11), [sorted by x-coordinate], "-|>"),

  edge((3.25, 10.2), (.19, 10.2), (.19, 10.67), "-|>"),
  
  blob((1.75, 11), [Uniformly distributed errors], tint: blue),

  edge("-|>"),

  blob((1.75, 12.25), [write in 'errors.txt'], tint: yellow),

  blob((.19, 10.67), [Midpoints], tint: blue),

  edge("-|>"),

  blob((.19, 12.25), [overwrite 'coord.txt'], tint: yellow),
)