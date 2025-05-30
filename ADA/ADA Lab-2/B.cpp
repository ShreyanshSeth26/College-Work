#include <iostream>
#include <vector>
using namespace std;
const int MOD = 1000000007;

void countSubsequences(vector<long> &arr, int target) {
    int i = 0;
    vector<long> newArr = arr;
    while (i < 1024) {
        newArr[i | target] = (newArr[i | target] + arr[i]);
        newArr[i | target] = newArr[i | target]% MOD;
        i++;
    }
    arr = newArr;
    
}

int main() {
    int queries;
    cin >> queries;
    vector<int> sequence;
    vector<long> arr(1024, 0);
    arr[0] = 1;

    while (queries--) {
        int mode, value;
        cin >> mode >> value;
        switch (mode) {
            case 1:
                countSubsequences(arr, value);
                break;
            case 2:
                cout<<arr[value]<<endl;
        }
    }
    return 0;
}