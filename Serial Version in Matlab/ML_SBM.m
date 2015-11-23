% This program is an implementation of the paper below with more adjustable parameters
% Chengbin Peng, Zhihua Zhang, Ka-Chun Wong, Xiangliang Zhang, David Keyes, "A scalable community detection algorithm for large graphs using stochastic block models", Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI), Buenos Aires, Argentina, 2015.


function [Z, B, Nc, Ni] = ML_SBM(W, K)
% W is the adjacency matrix
% K is the number of clusters (default 0)
% Z is the cluster indicator matrix (Z_{ik} = 1 means node i belongs to cluster k)
% B is the edge probability matrix

% example:
% W = sparse(rand(1000,1000) > 0.9); W(1:500, 501:1000) = 0; W(501:1000, 1:500) = 0; % generate a random graph
% [Z, B] = ML_SBM(W, 0);

dampB = 0.8; % ratio for old B
NCMIN = 20; % minimal cluster size, default 20
minLgap = 0; % minimal likelihood gap for merging, default 0;  0 or larger (for example, 10, 100)
maxIter = 50;
maxStageId = 10;

Wtrans = W';


N = size(W,1);
totalEdges = sum(sum(W,1));

isTrueK = 1;
if (K == 0)
    isTrueK = 0;
    K = round(N/10);
else
    Kori = K;
    K = K * 10;
end
if (K < 1)
    error(['Wrong number of clusters: K = ' num2str(K)]);    
end



isRandomInit = 1;









Zidx = repmat([1:K]', [ ceil(N/K), 1]);
Zidx = Zidx(1:N);
Zidx = Zidx(randperm(N));
Z = sparse(1:N, Zidx, 1, N, K, N);
Zinit = Z;





B = zeros(1,K+1);
B(end) = sum(sum(W,1))/N^2;
B(1:K) = B(end) + (1-B(end))*(ones(1,K)*0.5+rand(1,K)*0.5);
Binit = B;

Bnew = zeros(size(B));



result = zeros(100,1);




B = Binit;
Z = Zinit;
Ztemp = Zinit;


 isFinishStage = 0; 

for stageId = 0:(maxStageId - 1) 

   
Nc = sum(Z,1)';
Ni = zeros(1,K)';

for i = 1:K
    Ni(i) = sum(sum(W(logical(Z(:,i)*Z(:,i)')),1));
end





    
    for iter = 1:maxIter
        
        
        
        
        
        
       
        
        Blog = log(B);
        Blog1mB = log(1-B);
        
        
        Blog(B<=realmin) = log(realmin);
        Blog1mB(B>=1-realmin) = log(realmin);
        
        
        
        
        
        tic
        
        flag = 0;
        if isRandomInit
            order = randperm(N);
        end
        
        nChange = 0;
        
        for iidx = 1:size(Z,1)
            if isRandomInit
                i = order(iidx);
            else
                i = iidx;
            end
            
            
            cClusterID = Zidx(logical(Wtrans(:,i)));
            
            
            vec = cClusterID;
            unq = unique(vec);
            count = histc(vec, unq);
            
            if length(unq) == 1 && unq(1)  == Zidx(i)
                
                
                continue; 
            else
                
                if length(unq) == 1
                    IX = unq;
                else
                    
                    Nc(Zidx(i)) = Nc(Zidx(i)) - 1;
                    Nc(unq) = Nc(unq) - count;
                    
                    
                    term1 = count'.*(Blog(unq)-Blog(end));
                    term2 = Nc(unq)'.*(Blog1mB(unq)-Blog1mB(end));

                    
                    obj = term1+term2;
                    obj = obj';
                    
                    Nc(Zidx(i)) = Nc(Zidx(i)) + 1;
                    Nc(unq) = Nc(unq) + count;
                    [maxobj,~] = max(obj(1:length(unq)));
                    unqIdx = find(obj(1:length(unq))==maxobj);
                    unqIdxSz = length(unqIdx);
                    if  unqIdxSz > 1
                        
                        
                        rnd = rand(1, unqIdxSz);
                        unqIdx = unqIdx(rnd == max(rnd));
                    end
                    IX = unq(unqIdx);
                    
                end
                
            end
            
            
            if Zidx(i)~= IX
                
                nChange = nChange + 1;
            end
            
            
            

                
                if Zidx(i)~= IX
                    
                    Nc(Zidx(i)) = Nc(Zidx(i)) - 1;
                    if sum(unq == Zidx(i)) == 1
                        Ni(Zidx(i)) = Ni(Zidx(i)) - 2*count(unq == Zidx(i));
                    end
                    Zidx(i) = IX;  
                    Nc(Zidx(i)) = Nc(Zidx(i)) + 1;
                    if sum(unq == Zidx(i)) == 1
                        Ni(Zidx(i)) = Ni(Zidx(i)) + 2*count(unq == Zidx(i));
                    end
                end
                
            
        end
        
        
        
        
        if iter == 1
            fprintf('Number of changed nodes in stage %d: ', stageId);
        end
        
        fprintf('%d ', nChange);
        
        
            Z = sparse(1:N, Zidx, 1, N, K, N);
        
        
            
        
        p = Ni'./(Nc'.^2+eps);
        
        q = (totalEdges - sum(Ni))/(N^2 - sum(Nc'.^2)+eps);
        Bnew = [p, q];
        B = dampB*B + (1-dampB)*Bnew;
        
        
        
        
        
        if (isFinishStage == 0 && nChange <= 1e-3*N) || (isFinishStage >= 0.5 && nChange <= 0)
            fprintf('\n');
            disp(['Z converged. (nChange = ' num2str(nChange) ')']);
            break;
        end
        
        
        

    end
    
    
    if ( isFinishStage >= 0.5)
        isFinishStage = 1;
    end

    
    
    if (stageId >= 0) 
        
        
        
            Blast = B;
            Zidxlast = Zidx;
        
            
            len0 = length(B)-1;
            
                
                
                
                
                
                
                
                
                NcM = Nc*Nc';
                ZM = Z'*W*Z;
                
                
                
                
                NcM2 = repmat(Nc,[1, K]);
                NcM2 = NcM2 + NcM2';
                NcM2 = NcM2.*NcM2; 
                ZM2 = repmat(diag(ZM), [1,K]);
                ZM2 = ZM2 + ZM2';
                ZM2 = (ZM + ZM') +ZM2;
                pTemp = ZM2./(NcM2 + eps)+eps;
                
                
                Lsame = ZM2.*log(pTemp) + (NcM2 - ZM2).*log(1-pTemp);
                
                pTemp = B(end)+eps;
                Ldiff = (ZM.*log(pTemp) + (NcM - ZM).*log(1-pTemp))*2;
                pTemp = Ni./(Nc.*Nc+eps)+eps;
                Ldiffv = Ni .* log(pTemp) + (Nc.*Nc - Ni).*log(1-pTemp);
                Ldiff = Ldiff + repmat(Ldiffv, [1,K]) + repmat(Ldiffv, [1,K])';
                
                
                
                
                
                
                
                Imat = Lsame - Ldiff;
                Imat = Imat - diag(diag(Imat)+inf);
                [C, I] = max(Imat, [], 2); 
                
                
                Iful = [(1:K)', I]; 
                
                if (stageId >= 0 || isFinishStage >= 0.5) 
                    Iful = Iful((Nc<NCMIN &  (sum(ZM,2) - diag(ZM)~=0))|(C>-minLgap),:); 
                    
                else
                    Iful = Iful(C>-minLgap,:);
                                        
                    
                    
                end
                Iadj = sparse(Iful(:,1),Iful(:,2), 1, K,K,K);
                Iadj = Iadj+Iadj';
                Ire = components(Iadj);
                Ire(Nc == 0) = Ire(find(Nc~=0,1)); 
                [C, ~, Ire] = unique(Ire);
                
                Zidx = Ire(Zidx); 
                
                
                B2 = -ones(size(B));
                for i = 1:K
                    if (B2(Ire(i)) < B(i) && Nc(i)>0)
                        B2(Ire(i)) = B(i);                    
                    end
                end
                B = [B2(1:max(Ire)), B(end)];




















                
               
            len1 = length(B)-1;
            disp(['Community number change is ' num2str(len1 - len0) '; Remaining number is ' num2str(len1) '.']);
            if (isFinishStage ~= 1) 
                if isTrueK == 0 
                    if (len0 == len1 ) 
                        
                        isFinishStage = 0.6;   
                        
                    end
                else
                    if (len0 == len1) 
                        
                        isFinishStage = 0.7;
                        if (len1 > Kori)
                            isFinishStage = 0.5;
                        end
                    end
                    if (len1 < Kori)
                        B = Blast; 
                        Zidx = Zidxlast;
                        isFinishStage = 0.5;
                    end
                end

            end
    
                
            K = length(B) - 1;            
            Z = sparse(1:N, Zidx, 1, N, K, N); 


             

            
            
        
    end
    
    if ((isFinishStage == 0) && (stageId == maxStageId - 2))
        if (isTrueK == 1)
            isFinishStage = 0.5;
        else
            isFinishStage = 0.6;
        end
    end
    
    if (isFinishStage == 0.5) 
        hist(Zidx, unique(Zidx));
    end
    
    
    if (isFinishStage == 0.5) 
                    
            
            [~, IX] = sort(sum(Z,1), 'ascend');
            
            IXmergeSz = length(IX) - Kori;
            IXmerge = IX(1:IXmergeSz);            
            IXremain = 1:size(Z,2);
            IXremain(IXmerge)=-1;
            Zidx = IXremain(Zidx);
            IXremainLabel = IXremain;
            IXremainLabel(IXmerge) = [];            
            Zidx(Zidx == -1) = IXremainLabel(ceil(rand(sum(Zidx==-1),1)*Kori));
            
            [~, ~, Zidx] = unique(Zidx);
       
            
            B(IXmerge) = [];
            [~, ~, Zidx] = unique(Zidx);
            Z = sparse(1:N, Zidx, 1, N, Kori, N);
            
            disp(['[Due to Presetting of K] Community number change is ' num2str(Kori-K) '; Remaining number is ' num2str(Kori) '.']);
            K = Kori;
        
        
        
    end
    
    
    
    if (isFinishStage == 1)
         break;
    end
end
Z = real(Z);


IX = find(sum(Z,2)>1);
if ~isempty(IX)
    error('check?');
end
for i = 1:length(IX)
    IX2 = find(Z(IX(i),:));
    IX2IX = ceil(rand(1)*length(IX2));
    Z(IX(i),:) = 0;
    Z(IX(i),IX2(IX2IX)) = 1;
end


end

function ccomp = components(adj)
%Author: Ignat Drozdov


if isadjacency(adj)
    mat = adj.Graph;
else
    mat = adj;
end


[n, m] = size(mat);
if n ~= m, error ('Adjacency matrix must be square'), end;


if ~all(diag(mat)) 
    [foo, p, bar, r] = dmperm(mat | speye(size(mat)));
else
    [foo, p, bar, r] = dmperm(mat);  
end


sizes = diff(r);
k = length(sizes);


ccomp = zeros(1, n);
ccomp(r(1:k)) = ones(1, k);
ccomp = cumsum(ccomp);

ccomp(p) = ccomp;

end

function tf = isadjacency(in)


tf = strcmp('adjacency', class(in));

end
