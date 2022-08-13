%% This file receive inputs from python and execute FINAL
function [row, col, time_final] = run_final(Arow, Acol, Adata, ...
                                            Brow, Bcol, Bdata, ...
                                            Lrow, Lcol, Ldata, ...
                                            Aattribs, Battribs, ...
                                            alpha, maxiter, tol)
  % add path
  addpath('./final/');

  % get total number of nodes in graph
  fprintf('size(Lrow) is %d\n', max(Lrow))
  fprintf('size(Lcol) is %d\n', max(Lcol))
  N = max(max(Lrow), max(Lcol));

  % construct sparse A,B,L
  st_trans = tic();
  A = logical(sparse(Arow, Acol, Adata));
  B = logical(sparse(Brow, Bcol, Bdata));

  % normalize initial alignment matrix
  Ldata = Ldata / sum(sum(Ldata));
  L = sparse(Lrow, Lcol, Ldata, max(Lrow), max(Lcol))';
  %L = sparse(Lrow, Lcol, Ldata, N, N)';
  trans_time = toc(st_trans);
  fprintf('Recovering sparse matrix L took %.2f sec.\n', trans_time);

  % run FINAL
  st_final = tic();
  S = FINAL(A, B, Aattribs, Battribs, {}, {}, L, alpha, maxiter, tol);
  S_full = full(S)
  save('save.mat', 'S_full')
  [M, ~] = greedy_match(S);
  [row, col] = find(M == 1);
  % [col, row] = find(M == 1);
  time_final = toc(st_final);
  fprintf('Total time used by final is %.2f sec.\n', time_final);
end