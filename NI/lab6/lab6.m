set(0, 'DefaultTextInterpreter', 'latex');
% Задание 1
% Формируем множество случайных точек
X = [0, 1.5; 0, 1.5];
clusters = 8;
points = 10;
deviation = 0.1;
randomDots = nngenc(X, clusters, points, deviation);

% создаем сеть
network = competlayer(8);
network = configure(network, randomDots);
view(network);

%% Обучаем её
network.trainParam.epochs = 50;
network = train(network, randomDots);

%% Проверяем качество обучения
dotsForCheck = 1.5 * rand(2, 5);
result = vec2ind(sim(network, dotsForCheck));

% Смотрим на результат
figure;
hold on;
grid on;
scatter(randomDots(1, :), randomDots(2, :), 5, [0 0 1], 'filled');
scatter(network.IW{1}(:, 1), network.IW{1}(:, 2), 5, [0 1 0], 'filled');
scatter(dotsForCheck(1, :), dotsForCheck(2, :), 5, [1 0 0], 'filled');

%% Задание 2
% Формируем множество случайных точек
X = [0, 1.5; 0, 1.5];
clusters = 8;
points = 10;
deviation = 0.1;
randomDots = nngenc(X, clusters, points, deviation);

% создаем сеть
network = newsom(X, [2 4]);
network = configure(network, X);
% Обучаем её
network.trainParam.epochs = 150;
%network.inputWeights{1,1}.learnParam.init_neighborhood = 3;
%network.inputWeights{1,1}.learnParam.steps = 100;
network = train(network, randomDots);

%% Проверяем качество обучения
dotsForCheck = 1.5 * rand(2, 5);
result = vec2ind(sim(network, dotsForCheck));

% Смотрим на результат
figure;
hold on;
grid on;
scatter(randomDots(1, :), randomDots(2, :), 5, [0 0 1], 'filled');
scatter(network.IW{1}(:, 1), network.IW{1}(:, 2), 5, [0 1 0], 'filled');
scatter(dotsForCheck(1, :), dotsForCheck(2, :), 5, [1 0 0], 'filled');
plotsom(network.IW{1, 1}, network.layers{1}.distances);

%% Задание 3
% Формируем множество случайных точек
T = -1.5 * ones(2, 20) + 3 * rand(2, 20);

figure;
hold on;
grid on;
plot(T(1,:), T(2,:), '-V', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g', 'MarkerSize', 7);

%% Cоздаем сеть
network = newsom(T, 20);
network = configure(network, T);

% Обучаем её
network.trainParam.epochs = 600;
network = train(network, T);

% Координаты городов и центры кластеров сгенерированные сетью
figure;
hold on;
plotsom(network.IW{1,1}, network.layers{1}.distances);
plot(T(1,:), T(2,:), 'V', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g', 'MarkerSize', 7);
grid on;

%% Задание 4
% Инициализируем входное множество и распределение по классам
P = [0    0.3 -1.3 1.2 -1.2 -0.5  0.7 -1.4 0.3 0.6  0.8 0.5;
     0.7 -1.3  0.8 0.1  0.9 -0.7 -1.5  0.5  0  0.6 -0.7 0.1];
T = [-1   -1   -1  -1   -1   -1   -1   -1   1   1   -1   1];

% Отображаем его
plotpv(P, max(0, T));

% Строим вектор индексов классов
Ti = T;
Ti(Ti == 1) = 2;
Ti(Ti == -1) = 1;
Ti = ind2vec(Ti);

%% Создаем нейросеть
portion = [nnz(T(T == -1)) nnz(T(T == 1))] / numel(T);
network = lvqnet(12, 0.1);
network = configure(network, P, Ti);

%network.IW{1,1}
%network.LW{2,1}

% Обучем нейросеть
network.trainParam.epochs = 300;
network = train(network, P, Ti);

%% Проверяем качество обучения
% Задаем сетку
[X,Y] = meshgrid([-1.5 : 0.1 : 1.5], [-1.5 : 0.1 : 1.5]);

% Получаем выход сети
result = sim(network, [X(:)'; Y(:)']);
result = vec2ind(result) - 1;

% Граффическая демонстрация
plotpv([X(:)'; Y(:)'], result);
point = findobj(gca,'type','line');
set(point,'Color','g');
hold on;
plotpv(P, max(0, T));
