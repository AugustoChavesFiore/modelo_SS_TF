const result = document.getElementById("result");
const form = document.getElementById("form");
const train = document.getElementById("train");
const loader = document.getElementById("loader");

let model;

const leanrLinear= async()=>{
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    const xs = tf.tensor2d([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6], [10, 1]);
    const ys = tf.tensor2d([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], [10, 1]);
    await model.fit(xs, ys, {epochs: 500});
    return model;

};

train.addEventListener("click", async()=>{
    loader.innerHTML = "Entrenando Modelo...(Esto puede tardar un poco)";
    model = await leanrLinear();
    loader.innerHTML = "Modelo Listo, formula y = 2x + 6";
});

form.addEventListener("submit", async (e)=>{
    e.preventDefault();
    const data = new FormData(form);
    const x = parseFloat(data.get("x"));
    result.classList.add("alert", "alert-primary");
    if(isNaN(x)){
        result.innerHTML = "Por favor ingrese un número válido";
        return;
    }
    if(!model){
        result.innerHTML = "Por favor entrena el modelo antes de hacer una predicción.";
        return;
    }
    const output = model.predict(tf.tensor2d([x], [1, 1]));
    result.innerHTML = `
        <p><span class="fw-bold">Resultado de predicción: </span>  ${output.dataSync()[0]} </p>
        <p><span class="fw-bold">Resultado real: </span>  ${2*x + 6}<p>
        <p><span class="fw-bold">Error predicción: </span>  ${Math.abs(2*x + 6 - output.dataSync()[0])} </p>
    `;

});