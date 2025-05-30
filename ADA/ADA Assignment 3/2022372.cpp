#include <iostream>

using namespace std;

int numOne;
int numTwo;
int temp;
string inputBuffer;

struct Node {
    int data;
    Node* next;

    Node(int data) : data(data), next(nullptr) {}
};

Node* headOne;
Node* headTwo;
Node* prevNode;
Node* finalNode;

void printList(Node* node) {
    while (node != nullptr) {
        std::cout << node->data << " ";
        node = node->next;
    }
}

void mergeArray(Node* head1, Node* head2) {
    Node* finalHead = nullptr;
    Node* prev = nullptr;


    while (head1 && head2) {
        if (head1->data <= head2->data) {

            if (finalHead == nullptr) {
                finalHead = head1;
            } else {
                prev->next = head1;
            }
            prev = head1;
            head1 = head1->next;
        } else {

            if (finalHead == nullptr) {
                finalHead = head2;
            } else {
                prev->next = head2;
            }
            prev = head2;
            head2 = head2->next;
        }
    }

    if (head1) {
        prev->next = head1;
    } else if (head2) {
        prev->next = head2;
    }

    finalNode = finalHead;
}

int main() {
    cin>>numOne;
    cin>>numTwo;
    getline(cin,inputBuffer);
    cin>>temp;

    headOne= new Node(temp);
    prevNode=headOne;

    for (int i = 0; i < numOne-1; ++i) {
        Node* tempNode;
        cin>>temp;
        tempNode=new Node(temp);
        prevNode->next=tempNode;
        prevNode=tempNode;
    }
    getline(cin,inputBuffer);

    cin>>temp;
    headTwo= new Node(temp);
    prevNode=headTwo;

    for (int i = 0; i < numTwo-1; ++i) {
        Node* tempNode;
        cin>>temp;
        tempNode=new Node(temp);
        prevNode->next=tempNode;
        prevNode=tempNode;
    }
    getline(cin,inputBuffer);

    mergeArray(headOne,headTwo);

    printList(finalNode);

    return 0;
}