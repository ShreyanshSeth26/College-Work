#include <iostream>
#include <vector>
#include <queue>
#include <limits.h>
#include <algorithm>
#include <utility>

using namespace std;

struct Box {
    int h, w, d;
    Box(int height, int width, int depth) : h(height), w(width), d(depth) {}
};

bool canFit(const Box &a, const Box &b) {
    int a_dims[3] = {a.h, a.w, a.d};
    int b_dims[3] = {b.h, b.w, b.d};
    sort(a_dims, a_dims + 3);
    sort(b_dims, b_dims + 3);
    // Check that every dimension of a is strictly less than the corresponding dimension of b
    return (a_dims[0] < b_dims[0]) && (a_dims[1] < b_dims[1]) && (a_dims[2] < b_dims[2]);
}

int bfs(int s, int t, vector<int>& parent, vector<vector<int>>& residualGraph) {
    fill(parent.begin(), parent.end(), -1);
    parent[s] = -2;
    queue<pair<int, int>> q;
    q.push({s, INT_MAX});

    while (!q.empty()) {
        int curr = q.front().first;
        int flow = q.front().second;
        q.pop();

        for (int next = 0; next < residualGraph.size(); next++) {
            if (parent[next] == -1 && residualGraph[curr][next]) {
                parent[next] = curr;
                int new_flow = min(flow, residualGraph[curr][next]);
                if (next == t)
                    return new_flow;
                q.push({next, new_flow});
            }
        }
    }
    return 0;
}

int fordFulkerson(vector<vector<int>>& graph, int s, int t) {
    vector<vector<int>> residualGraph = graph;
    vector<int> parent(graph.size());
    int max_flow = 0;

    int flow;
    while ((flow = bfs(s, t, parent, residualGraph)) != 0) {
        max_flow += flow;
        int curr = t;
        while (curr != s) {
            int prev = parent[curr];
            residualGraph[prev][curr] -= flow;
            residualGraph[curr][prev] += flow;
            curr = prev;
        }
    }

    return max_flow;
}

int main() {
    int n;
    cin >> n;
    vector<Box> boxes;
    for (int i = 0; i < n; ++i) {
        int h, w, d;
        cin >> h >> w >> d;
        boxes.emplace_back(h, w, d);
    }

    int V = n + 2;
    int s = n, t = n + 1;
    vector<vector<int>> graph(V, vector<int>(V, 0));

    for (int i = 0; i < n; ++i) {
        graph[s][i] = 1; // Connect source to all boxes
        for (int j = 0; j < n; ++j) {
            if (i != j && canFit(boxes[i], boxes[j])) {
                graph[i][j] = 1; // Connect box i to box j if i can fit into j
            }
        }
    }

    // Connect boxes to sink if they have no incoming edges
    for (int j = 0; j < n; ++j) {
        bool has_incoming = false;
        for (int i = 0; i < n; ++i) {
            if (graph[i][j] == 1) {
                has_incoming = true;
                break;
            }
        }
        if (!has_incoming) {
            graph[j][t] = 1; // Connect box j to sink if no other box can fit into it
        }
    }

    int max_flow = fordFulkerson(graph, s, t);
    cout << "Minimum number of visible boxes: " << max_flow << endl;

    return 0;
}
