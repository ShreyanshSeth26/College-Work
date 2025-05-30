#include <iostream>
#include <vector>
#include <algorithm>
#include <stack>
using namespace std;

void dfsTopoSort(int v, const vector<vector<int>>& graph, vector<bool>& visited, stack<int>& topoStack) {
    visited[v] = true;
    for (int u : graph[v]) {
        if (!visited[u]) {
            dfsTopoSort(u, graph, visited, topoStack);
        }
    }
    topoStack.push(v);
}

void countPaths(const vector<vector<int>>& graph, vector<long long>& paths, stack<int>& topoStack) {
    while (!topoStack.empty()) {
        int v = topoStack.top();
        topoStack.pop();
        for (int u : graph[v]) {
            paths[u] += paths[v];
        }
    }
}

int main() {
    int n, m, s, t;

    cout << "Input Number Of Vertices: ";
    cin >> n;
    cout << "Input Number Of Edges: ";
    cin >> m;
    cout << "Input Starting Vertex: ";
    cin >> s;
    cout << "Input Target Vertex: ";
    cin >> t;

    vector<vector<int>> graph(n + 1), reverseGraph(n + 1);
    vector<bool> visited(n + 1, false);
    stack<int> topoStack;

    for (int i = 0; i < m; ++i) {
        int u, v;
        cout << "Input edge " << i + 1 << ": ";
        cin >> u >> v;
        graph[u].push_back(v);
        reverseGraph[v].push_back(u);
    }

    for (int i = 1; i <= n; ++i) {
        if (!visited[i]) {
            dfsTopoSort(i, graph, visited, topoStack);
        }
    }

    vector<long long> pathsFromS(n + 1, 0);
    pathsFromS[s] = 1;
    countPaths(graph, pathsFromS, topoStack);

    fill(visited.begin(), visited.end(), false);
    while (!topoStack.empty()) topoStack.pop();

    for (int i = 1; i <= n; ++i) {
        if (!visited[i]) {
            dfsTopoSort(i, reverseGraph, visited, topoStack);
        }
    }

    vector<long long> pathsToT(n + 1, 0);
    pathsToT[t] = 1;
    countPaths(reverseGraph, pathsToT, topoStack);

    vector<int> stCutVertices;
    for (int v = 1; v <= n; ++v) {
        if (v != s && v != t && pathsFromS[v] * pathsToT[v] == pathsFromS[t]) {
            stCutVertices.push_back(v);
        }
    }

    cout << "The (s,t)-cut vertices are: ";
    for (size_t i = 0; i < stCutVertices.size(); ++i) {
        if (i > 0) {
            cout << ", ";
        }
        cout << stCutVertices[i];
    }
    cout << endl;

    return 0;
}