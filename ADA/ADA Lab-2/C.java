import java.util.*;

public class C {
    private List<List<Integer>> adjacencyList;
    private int[] vertexScore;
    private int[] maxBeautyScore;
    
    public C(int numVertices) {
        adjacencyList = new ArrayList<>();
        for (int i = 0; i <= numVertices; i++) {
            adjacencyList.add(new ArrayList<>());
        }
        vertexScore = new int[numVertices + 1];
        maxBeautyScore = new int[numVertices + 1];
        Arrays.fill(vertexScore, 1); // Initialize scores with 1
    }
    
    public void addEdge(int start, int end) {
        adjacencyList.get(start).add(end);
        adjacencyList.get(end).add(start);
    }
    
    public int calculateMaxBeauty() {
        computeScores();
        Arrays.sort(maxBeautyScore); // Sort to find the max value at the end
        return maxBeautyScore[maxBeautyScore.length - 1]; // Return the last (max) element
    }
    
    private void computeScores() {
        maxBeautyScore[1] = adjacencyList.get(1).size() * vertexScore[1];
        for (int i = 2; i < adjacencyList.size(); i++) {
            int currentMax = vertexScore[i];
            for (int neighbor : adjacencyList.get(i)) {
                if (neighbor < i) {
                    vertexScore[i] = Math.max(currentMax + vertexScore[neighbor], vertexScore[i]);
                }
            }
            maxBeautyScore[i] = adjacencyList.get(i).size() * vertexScore[i];
        }
    }
    
    public static void main(String[] args) {
        @SuppressWarnings("resource")
        Scanner scanner = new Scanner(System.in);
        int totalVertices = scanner.nextInt();
        int totalEdges = scanner.nextInt();
        
        C graph = new C(totalVertices);
        while (totalEdges-- > 0) {
            int u = scanner.nextInt();
            int v = scanner.nextInt();
            graph.addEdge(u, v);
        }
        
        System.out.println(graph.calculateMaxBeauty()); // Output the maximum beauty
    }
}