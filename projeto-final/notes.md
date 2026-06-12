Perguntas:

(1) O trabalho é suficientemente complexo?
(2) Alterar o funcionamento do código para que os dados estejam sempre desde o começo na GPU é uma boa ideia? Nesse caso os dados só estarão no host no momento em que forem necessários.
(3) Alterar o código para que a impressão dos pesos da rede neural seja feita pela GPU é uma boa ideia?
(4) Buscar o uso de funções para otimização do código é algo que pode deixar o trazer uma boa complexidade ao trabalho? Por exemplo usar a função "cudaMallocPitch".
(5) Talvez a função "costDenseNetwork" possa ser otimizada utilizando GPU (na linha 119)
(6) Talvez a função "genUniformDenseNetwork" possa ser otimizada utilizando GPU (na linha )