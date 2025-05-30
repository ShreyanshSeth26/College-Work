#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
using namespace std;

// Count Minimum Containers Needed
int cntMin(int N, vector<int>& juice, vector<int>& cap) {
    vector<int> capSort = cap;
    sort(capSort.begin(), capSort.end());
    int totalJuice = accumulate(juice.begin(), juice.end(), 0);
    int cnt = 0, filled = 0;
    for (int i = N - 1; i >= 0 && filled < totalJuice; --i) {
        filled += capSort[i];
        cnt++;
    }
    return cnt;
}

// Minimize Effort for Juice Transfer And Run Minimization Process
void optJuice(int N, vector<int>& juice, vector<int>& cap) {
    int totalJuice = accumulate(juice.begin(), juice.end(), 0);
    int totalCap = accumulate(cap.begin(), cap.end(), 0);
    int minC = cntMin(N, juice, cap);
    vector<vector<vector<int>>> dp(N + 1, vector<vector<int>>(totalCap + 1, vector<int>(minC + 1, static_cast<int>(1e9))));
    dp[0][0][0] = 0;
    for (int i = 0; i < N; ++i) {
        for (int soFar = 0; soFar <= totalCap; ++soFar) {
            for (int cnt = 0; cnt <= minC; ++cnt) {
                if (cnt > 0 && soFar >= cap[i]) {
                    dp[i + 1][soFar][cnt] = min(dp[i + 1][soFar][cnt], dp[i][soFar - cap[i]][cnt - 1]);
                }
                dp[i + 1][soFar][cnt] = min(dp[i + 1][soFar][cnt], dp[i][soFar][cnt] + juice[i]);
            }
        }
    }
    int minE = static_cast<int>(1e9);
    for (int soFar = totalJuice; soFar <= totalCap; ++soFar) {
        minE = min(minE, dp[N][soFar][minC]);
    }
    cout << minC << " " << minE << endl;
}

//Run The code
int main() {
    int N;
    cin >> N;
    vector<int> juice(N), cap(N);
    for (int& j : juice) cin >> j;
    for (int& c : cap) cin >> c;
    optJuice(N, juice, cap);
    return 0;
}