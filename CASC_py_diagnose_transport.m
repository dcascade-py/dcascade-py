sed_range = [min(C.psi) max(C.psi)]

QBi_trsum = squeeze(sum(C.Qbi_tr,2,'omitnan')); %sum across FIRST dim of QB_tr_perc, which is the 'from' index of sediment? how to check?
QBi_mobsum = squeeze(sum(C.Qbi_mob,2,'omitnan')); %sum across FIRST dim of QB_tr_perc, which is the 'from' index of sediment? how to check?

%Qbi_dep will have a shape of (dep_save_number, n_reaches, 1, n_classes + 1)
%so, there's an extra integer in the n_classes dimension\
%%
% (C.Qbi_dep{10,4})
%%
%I'd like to plot the Xth percentile for now
qperc = 90
[foo, qsi] = sort(Qbry) %sorted indices
Q_at_qperc = prctile(Qbry,qperc)
sorted_index_at_qperc = find(foo>=Q_at_qperc,1,'first')
Qbry(qsi(sorted_index_at_qperc))

Qcmap = colormap(jet(length(qranges)))
xStrings = string(dmi(2:2:end));

yax = ([-1e5 1e5])*10
stlims = [0.05 .04];
for t = qsi(sorted_index_at_qperc)

   % if(Qbry(t)<120),continue,end %let's not watch a bunch of low flow

    figure(1001),clf

    
    %sand went in at reach 3?
    rrlist =[4 6 7 8 9 10]
    
    for r=1:length(rrlist)  %reach number
        rr = rrlist(r);

        %transport THROUGH, from above and this reach
        
%area plots of Qb transported
        subplot_tight(6,5,5*r-5+1,stlims)
        %ah1 = area(psi,sq(QBi_trsum(t,rr,:,:))); hold on
        %ah1 = area(psi,sq(C.transported_class(:,t,rr))); hold on
        ah1 = area(psi,sq(transp_class_sum(:,rr,:))' ); hold on

        ylim(max(0,yax))
        xlabel(['reach #' num2str(rr)])

       % ylim([0 200])
        if r==1,title('Transported'),end
        set(gca,'xtick',psi(2:2:end),'xticklabel',xStrings), xlim(sed_range),

        %area plots of Qb mobilized and deposited
        subplot_tight(6,5,5*r-5+2,stlims)
        %QBi_depsum = (C.Qbi_dep{t,r}(1,2:end))/1e3
        ah2 = area(psi,sq(mobil_class_sum(:,rr,:))' ); hold on
        ah3 = area(psi,-sq(dep_class_sum(:,rr,:))' ); hold on
        
        %ylim([-10 10]*50)
        if r==1,title('Mob/Dep'),end
        set(gca,'xtick',psi(2:2:end),'xticklabel',xStrings), xlim(sed_range),
        for pl = 1:length(ah2) %colors - red is high flows. 
            ah1(pl).FaceColor = Qcmap(pl,:); ah1(pl).LineStyle = 'none'
            ah2(pl).FaceColor = Qcmap(pl,:); ah2(pl).LineStyle = 'none'
            ah3(pl).FaceColor = Qcmap(pl,:); ah3(pl).LineStyle = 'none'
        end
        ylim([-5e5 5e5])

%line plots of Fi
        subplot_tight(6,5,5*r-5+3,stlims)
        Fi_mobilised = [C.mobilised_class(:,t,rr)./sum(C.mobilised_class(:,t,rr))]';
        plot(psi,Fi_mobilised  )

        hold on

        Fi_transported = [C.transported_class(:,t,rr)./sum(C.transported_class(:,t,rr))]';
        plot(psi,Fi_transported  )

        Fi_deposited= [C.deposited_class(:,end,rr)./sum(C.deposited_class(:,end,rr))]';
        plot(psi,Fi_deposited  ) %NOTE: not identicalto Fi_r_ac
        % 
        Fi_tot= [C.tot_sed_class(:,end,rr)./sum(C.tot_sed_class(:,end,rr))]';
         plot(psi,Fi_tot ,'g:','linew',2 ) %NOTE: Identical as it should be to Fi_r_ac

        
        %
        plot(psi,squeeze(C.Fi_r_ac(t,rr,:)),'k--')
        ylim([0 .8]), grid on
        %set(gcf,'pos',[ 1230          70         358         210]) %home laptop
        %set(gcf,'pos',[ -2361         436         866         362]) %work
        %plot(psi,Qbidiag ./ sum(Qbidiag),'k-','linew',1.5)
        %legend('Fi_Mobil','Fi_Transported','Fi_Deposited','Fi_active')
        legend('FiMobil','FiTransported','FiDeposited','Fitot','Fi-active') %chose not to save deposited? need to make that save more efficient. maybe don't save if Q low.
        
        set(gca,'xtick',psi(2:2:end),'xticklabel',xStrings), xlim(sed_range),

        %q vs stage
        %area plots of Qb mobilized and deposited
        subplot_tight(6,5,5*r-5+4,stlims)
        plot(Qbry(1:end-1),C.flow_depth((1:end-1),rr),'k.'), hold on
        plot(Qbry(1:end-1),C.Q(1:end-1,rr)./(C.flow_depth(1:end-1,rr) .* C.wac(1:end-1,rr)),'r.')
        grid on
        axis([0 1000 0 2.5])
        %q vs tau


    end

title(['Days:'  num2str(C.Tdays(t)) 'Q=' num2str(Qbry(t))])
    pause(.1)
    
end

%%
% 
% figure(1002)
% 
% imagesc(reachlen,C.Tdays(1:tmaxindex),C.D50_AL(1:tmaxindex,:)-C.D50_AL(1,:)), shading flat

%%
porosity = 0.4  %phi in the model, confusing name

%Qbi_input[:,0,6].sum() = 25003
%Qbi_input[:,0,5:7].sum()
%Out[22]: 50006.293806

%THIS IS deposit_layer in the input script. 
%only works if saving every step  C.tot_sed(1,1) / C.Length(1)
disp('broken code if not saving every timestep')

%and same
sum(C.tot_sed_class(:,1,1)) / C.Length(1)
sum(C.deposited_class(:,1,1)) / C.Length(1)
sum(C.mobilised_class(:,1,1)) / C.Length(1)
sum(C.transported_class(:,1,1)) / C.Length(1) %nothing transported in 1 as its top reach, could go look for inputs though. 
sum(C.transported_class(:,1,2)) / C.Length(2) %this is the second reach, so we can see transported(2) similar to mobilised(1)

% only works if saving every step C.tot_sed(end,1) / C.Length(1) %meters
sedclass=7

volchange = squeeze(C.deposited_class(sedclass,:,1:14)) - (squeeze(C.deposited_class(sedclass,1,1:14)))';

figure(1003),clf
%tracks volume increases in the domain
%plot(C.Tdays(1:end-1),volchange(1:end-1,:)), hold on
plot(C.Tdays(1:365:end-366),volchange(1:end-1,:)), hold on
%pretty sure: stracks which reach an outlet-exiting cascade came from, so it's all in 1, not 9 basically
plot(C.Tdays(1:end-1),cumsum(sum(C.Q_out(1:end-1,:,sedclass),[2 3])),'k--') 


ylabel('Vol Change m^3'),xlabel('days')
%this does not seem to balance. Does mobilized material get added so than it can then be deposited?
%in:
25003 
%on bed: 
sum((volchange(end,:))) 
%out:
% figure, plot(C.Tdays(1:end),sum(C.Q_out(:,:,sedclass),[2]))
all_out = sum(C.Q_out(:,:,sedclass),[1 2])

%what moved:
all_tr = sum(C.transported_class(sedclass,:,:),[2 3])
%what was eroded:
all_erod = sum(C.mobilised_class(sedclass,:,:),[2 3])

excess_dep = sum((volchange(end,:)))  - 25003

%These seem close. 
excess_dep - (all_tr - all_out) 
all_out

