alert("asdf")

let DATA = new Array();

d3.csv("example_data/0/visor/spectra/cumulative_ages.csv", function (data) {
    let transformed = Object.values(data).map(a => +a);
    DATA.push(transformed)
    // DATA.push(data.forEach(e => e.forEach(e2 => +e2)));
}).then(function () {
    console.log(DATA);
    main(d3.transpose(DATA));

})



function main(data) {
    let svg = d3.select("#container > svg");

    let r = 5;

    let scaleX = d3.scaleLinear()
        .domain([0, 50])
        .range([0 + r, 500 - r]);

    let maxData = Math.max(...data.map(array => Math.max(...array)));

    console.log(maxData);

    let scaleY = d3.scaleLinear()
        .domain([0, maxData])
        .range([500 - r, 0 - r]);

    svg
        .append("g")
        .selectAll("dot")
        .data(data)
        .enter()
        .append("circle")
        .attr("cx", function (d, i) { return scaleX(i) })
        .attr("cy", function (d, i) { return scaleY(d[0]) })
        .attr("r", r)
        .attr("fill", "#69b3a2")


}