import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs){
    const model = tf.sequential()
    // Primeira camada da rede
    // entrada de 7 posiçoes (idade normalizada, +3 cores, + localizaçoes)

    // 80 neuronios = aqui coloquei tudo isso, pq tem pouca base de treino
    // quanto mais neuronios, mas complexidade a rede pode aprender
    // e consequentemente, mais processamento ela vai usar

    // A ReLU age como um filtro
    // E como se ela deixasse somente os dados interessantes seguirem viagem na rede
    /// Se a informaçao chegou nesse neuronio é positiva, passa pra frente!
    ///  se foor zero ou negativa, pode jogar fora, nao vai servir para nada
    model.add(tf.layers.dense({inputShape:[7], units: 80, activation:'relu'}))


    //Saida: 3 neuronios
    // um para cada categoria(premium, medium, basic)
    // activation: softmax normaliza a saida em probabilidades
    model.add(tf.layers.dense({units:3, activation:'softmax'}))

    //Compilando o modelo
    // optimizer Adam (Adaptive Moment estimation)
    // e um treinador pessoal moderno para redes neurais:
    // ajusta os pesos de forma eficiente e inteligente
    // aprender com historico de errors e acertos

    //loss: categoricalCrossentropy
    // Ele comparar o que o modelo "acha" (os scores de cada categoria)
    // com a resposta certa
    // a categoria premium sera sempre [1,0,0]

    //quanto mais distrande da previsao do modelo da resposta correta
    // maior o erro (loss)
    //Examplo classico: classificaçao de imagens, recomendaçao, categorizaçao de usuario
    //qualquer coisa em que a resposta certa é "apenas uma entre varias possiveis"

    model.compile({
        optimizer:'adam', 
        loss:'categoricalCrossentropy', 
        metrics:['accuracy']
    })

    //Treinamento do modelo
    //verbose: desabilita o log interno ( e usa so call back)
    // epochs: quantidade de veces que vai rodar no dataset
    // shuffle: embaralha os dados, para evitar vies
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,
            epochs:100,
            shuffle: true,
            callbacks:{
                //onEpochEnd: (epoch, log) => console.log(
                //    `Epoch: ${epoch}: loss = ${log.loss}`
                //)
            }
        }
    )

    return model
}

async function predict(model, pessoa) {
    //transformar o array js para o tensor(tfjs)
    const tfInput = tf.tensor2d(pessoa)

    // Faz a prediçao (Putput sera um vetor de 3 probabilidades)

    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    return predArray[0].map((prob, index)=> ({prob, index}))
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

//inputXs.print();
//outputYs.print();


// quanto mais dado melhor
// assim o algoritmo consegue entender melhor os padroes comoplexos
// dos dados
const model = await trainModel(inputXs, outputYs)

const pessoa = {nome:'Ze', idade :28, cor: 'Verde', localizacao:"Curitiba"}
//normalizando a idade da nova pessoa usando o mesmo padrado do treino
//Exemplo: idadde_min=25, idade_max = 40, entao (28-25)/(40-25) = 0.2

const pessoaTensorNormalizado = [
    [
        0.2, //idade normalizada
        1, // cor azul
        0, // cor vermelho
        0, // cor verde
        0, // localizaçao sp
        1, // localizaçao rio
        0 // localizaçao cwb
    ]
]

const predictions = await predict(model, pessoaTensorNormalizado)
const results = predictions.sort((a,b) => b.prob- a.prob).map( p => `${labelsNomes[p.index]} (${(p.prob*100).toFixed(2)}%)`).join('\n')
console.log(results)