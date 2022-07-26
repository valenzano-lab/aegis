let PARAMS;
let CATEGS;

fetch("./data.json")
  .then((response) => response.json())
  .then((data) => main(data));

function setup() {
  // Make submit button respond to click
  d3.select("input.submit").on("click", function () {
    let inputParams = {};
    d3.selectAll(".input-param").each(function () {
      inputParams[this.name] = this.value;
    });
    console.log(inputParams);
  });
}

function main(iData) {
  setup();

  //   Extract data
  let categs = iData["categories"];
  let params = iData["parameters"];

  //   Add category labels
  d3.select(".option-container")
    .selectAll()
    .data(categs)
    .enter()
    .append("div")
    .attr("class", (d) => `options ${d.category}`)
    .append("p")
    .attr("class", "options-label")
    .text((d) => d.category);

  // Add parameter boxes and labels
  for (const datum of params) {
    let div = d3
      .select(`div.options.${datum.category}`)
      .append("div")
      .attr("class", `option ${datum.name}`);

    div.append("p").text(datum.nick);
    div
      .append("p")
      .text(datum.name)
      .style("font-size", "0.8rem")
      .style("font-family", "monospace");
    div.append("p").text(datum.description).style("font-size", "0.8rem");
    let inputDiv = div.append("div");

    switch (datum.type) {
      case undefined:
        inputDiv
          .append("input")
          .attr("class", "input-param")
          .attr("name", datum.name)
          .attr("value", datum.default);
        break;
      case "bool":
        inputDiv
          .append("input")
          .attr("type", "checkbox")
          .attr("id", datum.name)
          .property("checked", datum.default == "true");
        break;
      case "radio":
        let select = inputDiv.append("select").attr("id", datum.name);
        //   .attr("size", datum["radio-options"].length);
        for (const o of datum["radio-options"]) {
          select
            .append("option")
            .attr("value", o)
            .text(o)
            .property("selected", o == datum["default"]);
        }
    }
  }

  //   Calculate the maximum number of generations and update the text
  let nGenerationsDiv = d3
    .select(".option.STAGES_PER_SIMULATION_")
    .append("div");
  d3.selectAll(".input-param").on("input change", function () {
    let stagesPerSimulation = d3
      .select(".input-param[name=STAGES_PER_SIMULATION_]")
      .property("value");

    let maturationAge = d3
      .select(".input-param[name=MATURATION_AGE]")
      .property("value");

    let nGenerations = Math.round(+stagesPerSimulation / +maturationAge);

    nGenerationsDiv.text(nGenerations);
  });
}
