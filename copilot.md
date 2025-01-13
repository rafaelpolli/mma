# Microsoft Copilot Studio

- Solução low code para criação agentes de texto.
> É possível criar um agente mesmo com pouco conhecimento de
> programação/ciência de dados porém é importante ressaltar que para
> utilização de forma completa e correta da ferramenta é necessário
> treinamento ou estudo da documentação e não dispensa um processo de
> avaliação e curadoria.
- Utiliza nativamente o GPT-4o para respostas generativas.
> Não é possível controlar parâmetros como temperatura, número de
> tokens ou até mesmo versão/snapshot do modelo, o que pode levar
> à mudanças no funcionamento.
- O Copilot disponível no módulo Customer Service Workspace do Dynamics é diferente e extremamente limitado, 
> Conta somente com utilização de uma base de conhecimento que deve ser
> inserida no editor da ferramenta e não possuí features básicas como
> utilização de prompts.
- Permite implantação (deploy) rápida de agentes no teams, dispensando infraestrutura e dependência de desenvolvimento de plataforma.
- É possível adicionar arquivos de diversos formatos, sites e sharepoints como base de conhecimento para que a IA Generativa use para gerar respostas. 
> Não há como selecionar estratégia de busca, segmentação ou quantidade de
> documentos retornados para uso da IA. O uso de sharepoint como base de
> conhecimento ainda necessita de permissão de DLP.
- É possível criar fluxo de conversas, ações como enviar um e-mail e fazer reconhecimento de entidades de maneira simples e gráfica.
- A ferramenta conta com um módulo de Analytics onde é possível monitorar o uso, engajamento, NPS, feedback, histórico de conversas, abandono e resolutividade.
> O módulo de Analytics é simples e não fornece o trecho da base de
> conhecimento utilizada para gerar determinada resposta, além disso
> para democratização do log de conversas no Mesh é necessário a
> configuração/desenvolvimento de um conector ou serviço para
> transferência de dados entre Azure e AWS.
- Através do Direct Line, pode-se utilizar um agente por meio de API REST.
> É necessário verificar se a licença adquirida permite publicação em
> Apps customizados/Apps mobiles ou somente via teams. Além disso é
> necessário configurar o método de autenticação.
- Muitas das features do Copilot Studio estão em estado de preview e não são aconselhados para uso em produção pois podem sofrer alterações ou parar de funcionar. (Ex. Orquestração de tópicos e ações por IA generativa, alguns conectores, ações, etc.)
> Mencionado pela Walkiria o uso e casting de variáveis que deixaram de
> funcionar.
