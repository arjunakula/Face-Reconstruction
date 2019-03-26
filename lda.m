function model = lda(X, y, num_components)
  dim = size(X,1);
  c = max(y); 
  
  if(nargin < 3)
    num_components = c - 1;
  end
  
  num_components = min(c-1,num_components);
  
  meanTotal = mean(X,2);
  
  Sw = zeros(dim, dim);
  Sb = zeros(dim, dim);
  for i=1:c
    Xi = X(:,find(y==i));
    meanClass = mean(Xi,2);
    % center data
    Xi = Xi - repmat(meanClass, 1, size(Xi,2));
    % calculate within-class scatter
    Sw = Sw + Xi*Xi';
    % calculate between-class scatter
    Sb = Sb + size(Xi,2)*(meanClass-meanTotal)*(meanClass-meanTotal)';
  end

  % solve the eigenvalue problem
  [V, D] = eig(Sb,Sw);
  
  % sort eigenvectors descending by eigenvalue
  [D,idx] = sort(diag(D), 1, 'descend');
  
  V = V(:,idx);
  % build model
  model.name = 'lda';
  model.num_components = num_components;
  model.D = D(1:num_components);
  model.W = V(:,1:num_components);
end
