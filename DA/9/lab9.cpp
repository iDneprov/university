#include <iostream>
#include <map>
#include <algorithm>
#include <vector>

std::map<int, std::vector<int> > g;
std::vector<bool> used;
std::vector<int>  mt;

// обход в губину
bool kuhn_dfs(int pos)
{
    if (used[pos])
        return false;

    used[pos] = true;

    for (auto to : g[pos])
        // если мы не посещали вершину или нашли увеличивающую цепочку, пишем её в текущую цепочку
        if (mt[to] == -1 || kuhn_dfs(mt[to])) {
            mt[to] = pos;
            mt[pos] = to;
            return true;
        }
    return false;
}

int main()
{
    int n, m, a, b;
    std::cin >> n >> m;
    used.assign(n, 0);
    // зачитываем граф
    while (std::cin >> a >> b) {
        --a;
        --b;
        g[a].push_back(b);
        g[b].push_back(a);
    }
    // сортируем зачитанные данные
    for (int i = 0; i < g.size(); ++i) {
        std::sort(g[i].begin(), g[i].end());
    }
    // инициилизируем пустое паросочетание
    mt.assign(n, -1);
    // ищем увеличивающую цепочку для каждой вершины
    for (int i = 0; i < n; ++i)
        if (mt[i] == -1) {
            used.assign(n, 0);
            kuhn_dfs(i);
        }
    // перегиняем ответ из текущей цепоки в map для вывода в отстортированном
    std::map<int, int> ans;
    for (int i = 0; i < n; ++i)
        if (mt[i] != -1)
            ans[std::min(i, mt[i])] = std::max(i, mt[i]);
    // выводим результат
    std::cout << ans.size() << std::endl;
    for (auto i : ans)
        std::cout << i.first + 1 << ' ' << i.second + 1 << std::endl;
    return 0;
}